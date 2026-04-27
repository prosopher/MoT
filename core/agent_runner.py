from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from core.agent import Agent, AgentGeneration, HubAgent, get_past_seq_len
from core.common import PastKeyValues, extract_past_key_values, read_json, set_seed
from core.context import Context
from core.eval_util import (
    InferenceProfiler,
    compute_generation_f1,
    get_squad_v11_dataset_spec,
    postprocess_generated_answer,
)
from core.topology import Edge, build_edge_map, build_nodes_and_edges
from core.config import resolve_device
from core.train_util import get_train_config_path


CACHE_MODE_RETAIN = "retain"
CACHE_MODE_FREE = "free"
SUPPORTED_CACHE_MODES = (CACHE_MODE_RETAIN, CACHE_MODE_FREE)


@dataclass
class AgentRunnerConfig:
    alg: str = "mot"
    checkpoint_dir_path: str = ""
    device: str = "auto"
    max_turns: int = 4
    generation_max_new_tokens: int = 48
    max_prompt_tokens: Optional[int] = None
    seed: int = 42
    log_turns: bool = True
    log_max_chars: int = 600
    cache_mode: str = CACHE_MODE_RETAIN


@dataclass
class AgentTurnRecord:
    agent_id: str
    prompt: str
    response: str
    raw_response: str
    stop_reason: str
    cache_seq_len_before: int
    cache_seq_len_after: int
    cache_mode: str = CACHE_MODE_RETAIN
    translated_from: Optional[str] = None
    translated_edge_id: Optional[str] = None
    translated_source_seq_len: int = 0
    translated_target_seq_len: int = 0
    translated_delta_tokens: int = 0
    offloaded_to: Optional[str] = None
    offload_edge_id: Optional[str] = None
    offload_source_seq_len: int = 0
    offload_target_seq_len: int = 0
    offload_delta_tokens: int = 0
    cache_cleared_after_turn: bool = False


@dataclass
class AgentRunnerResult:
    question: str
    gold_answers: List[str]
    prediction: str
    f1: float
    transcript: str
    turns: List[AgentTurnRecord]
    profile: Dict[str, Optional[float]]

    @property
    def peak_memory_gib(self) -> float:
        peak = self.profile.get("peak_memory_bytes")
        if peak is None:
            return float("nan")
        return float(peak) / (1024 ** 3)


class KVCacheTranslationAdapter:
    """Dispatches full-prefix cache replay/translation to each implemented algorithm."""

    def __init__(self, *, ctx: Context, translator_pool, alg: str) -> None:
        self.ctx = ctx
        self.translator_pool = translator_pool
        self.alg = alg
        self.edge_map = build_edge_map(ctx.edges)
        self._sent_source_seq_lens: Dict[Tuple[str, str], int] = {}

    def _get_edge(self, src_node_id: str, tgt_node_id: str) -> Edge:
        edge_id = f"{src_node_id}_to_{tgt_node_id}"
        edge = self.edge_map.get(edge_id)
        if edge is None:
            raise ValueError(
                f"Missing translator edge {edge_id!r}. AgentRunner requires both A_to_B and B_to_A "
                f"for two-agent conversation. Available edges: {sorted(self.edge_map)}"
            )
        return edge

    def _shared_prefix_ids_for_edge(self, target_agent: Agent, transcript_text: str) -> torch.Tensor:
        # Existing C2C/LSC/KVComm/MoT eval paths tokenize an edge prefix with the
        # target tokenizer and feed the same ids to the source model. Keep that
        # convention here so the runner matches checkpoint-time assumptions.
        return target_agent.encode_text(transcript_text)

    @torch.inference_mode()
    def translate_full_cache(
        self,
        *,
        source_agent: Agent,
        target_agent: Agent,
        transcript_text: str,
    ) -> Tuple[PastKeyValues, Dict[str, Any]]:
        edge = self._get_edge(source_agent.node_id, target_agent.node_id)
        edge_id = edge.id
        alg = self.alg
        target_input_ids = target_agent.encode_text(transcript_text)

        if alg == "interlat":
            train_mod = importlib.import_module("interlat.train")
            source_input_ids = source_agent.encode_text(transcript_text)
            source_hidden = train_mod.extract_last_hidden_states(
                self.ctx.mm.get_model(edge.src_id),
                source_input_ids,
            )
            translated_latents = self.translator_pool.translate_hidden_states(
                edge_id=edge_id,
                source_hidden_states=source_hidden,
            )
            translated_past = train_mod.build_latent_conditioned_past(
                self.ctx.mm.get_model(edge.tgt_id),
                prefix_input_ids=target_input_ids,
                latent_prefix=translated_latents,
            )
            source_past_for_delta = extract_past_key_values(
                self.ctx.mm.get_model(edge.src_id),
                source_input_ids,
            )
        else:
            shared_input_ids = self._shared_prefix_ids_for_edge(target_agent, transcript_text)
            source_past = extract_past_key_values(self.ctx.mm.get_model(edge.src_id), shared_input_ids)
            source_past_for_delta = source_past

            if alg == "c2c":
                train_mod = importlib.import_module("c2c.train")
                from core.common import replace_top_layers

                native_target_past = extract_past_key_values(self.ctx.mm.get_model(edge.tgt_id), shared_input_ids)
                translated_top_past = train_mod.translate_top_layers(
                    translator_pool=self.translator_pool,
                    train_config=self.ctx.config,
                    sharer_past_key_values=source_past,
                    receiver_past_key_values=native_target_past,
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=self.ctx.mm.get_model_spec(edge.tgt_id),
                )
                translated_past = replace_top_layers(
                    base_past_key_values=native_target_past,
                    translated_top_past_key_values=translated_top_past,
                )
            elif alg == "lsc":
                translated_past = self.translator_pool.translate_layers(
                    past_key_values=source_past,
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=self.ctx.mm.get_model_spec(edge.tgt_id),
                )
            elif alg == "kvcomm":
                translated_past = self.translator_pool.build_replayed_target_past(
                    edge_id=edge_id,
                    source_past_key_values=source_past,
                )
            elif alg == "mot":
                translated_past, _ = self.translator_pool.build_replayed_target_past(
                    source_past_key_values=source_past,
                    prefix_input_ids=shared_input_ids,
                    target_model=self.ctx.mm.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=self.ctx.mm.get_model_spec(edge.tgt_id),
                )
            elif alg == "mot-h":
                train_mod = importlib.import_module("mot-h.train")
                source_past_h, hidden_states = train_mod.extract_model_prefill_artifacts(
                    self.ctx.mm.get_model(edge.src_id),
                    shared_input_ids,
                )
                source_past_for_delta = source_past_h
                source_canonical_attn_input_block = train_mod.extract_selected_layer_canonical_attn_input_block(
                    self.ctx.mm.get_model(edge.src_id),
                    hidden_states,
                    self.ctx.cm.get_src_layer_indices(edge_id),
                )
                translated_past, _ = self.translator_pool.build_replayed_target_past(
                    source_past_key_values=source_past_h,
                    prefix_input_ids=shared_input_ids,
                    target_model=self.ctx.mm.get_model(edge.tgt_id),
                    src_node_id=edge.src_id,
                    tgt_node_id=edge.tgt_id,
                    tgt_spec=self.ctx.mm.get_model_spec(edge.tgt_id),
                    source_canonical_attn_input_block=source_canonical_attn_input_block,
                )
            else:
                raise ValueError(f"Unsupported AgentRunner alg={alg!r}")

        source_seq_len = get_past_seq_len(source_past_for_delta)
        previous_seq_len = self._sent_source_seq_lens.get((edge.src_id, edge.tgt_id), 0)
        delta_tokens = max(0, source_seq_len - previous_seq_len)
        self._sent_source_seq_lens[(edge.src_id, edge.tgt_id)] = source_seq_len
        return translated_past, {
            "edge_id": edge_id,
            "source_seq_len": source_seq_len,
            "target_seq_len": get_past_seq_len(translated_past),
            "delta_tokens": delta_tokens,
        }


class AgentRunner:
    def __init__(
        self,
        *,
        ctx: Context,
        translator_pool,
        alg: str = "mot",
        max_turns: int = 4,
        generation_max_new_tokens: int = 48,
        max_prompt_tokens: Optional[int] = None,
        seed: int = 42,
        log_turns: bool = True,
        log_max_chars: int = 600,
        cache_mode: str = CACHE_MODE_RETAIN,
    ) -> None:
        if len(ctx.nodes) < 2:
            raise ValueError("AgentRunner requires at least two nodes in the checkpoint model pool.")
        if cache_mode not in SUPPORTED_CACHE_MODES:
            raise ValueError(f"Unsupported cache_mode={cache_mode!r}; expected one of {SUPPORTED_CACHE_MODES}")
        self.ctx = ctx
        self.translator_pool = translator_pool
        self.alg = alg
        self.max_turns = int(max_turns)
        self.generation_max_new_tokens = int(generation_max_new_tokens)
        self.max_prompt_tokens = max_prompt_tokens
        self.seed = int(seed)
        self.log_turns = bool(log_turns)
        self.log_max_chars = max(80, int(log_max_chars))
        self.cache_mode = cache_mode
        self.device = ctx.config.device
        self.profiler = InferenceProfiler(self.device)
        self.cache_translator = KVCacheTranslationAdapter(ctx=ctx, translator_pool=translator_pool, alg=alg)

        node_a = next((node for node in ctx.nodes if node.id == "A"), ctx.nodes[0])
        node_b = next((node for node in ctx.nodes if node.id == "B"), ctx.nodes[1])
        stop_sequences = ("\nAgent A:", "\nAgent B:", "\nQuestion:", "\nPassage:")
        self.agent_a = HubAgent(
            node_id=node_a.id,
            model=ctx.mm.get_model(node_a.id),
            tokenizer=ctx.mm.get_tokenizer(node_a.id),
            device=self.device,
            max_new_tokens=self.generation_max_new_tokens,
            stop_sequences=stop_sequences,
            max_prompt_tokens=max_prompt_tokens,
        )
        self.agent_b = Agent(
            node_id=node_b.id,
            model=ctx.mm.get_model(node_b.id),
            tokenizer=ctx.mm.get_tokenizer(node_b.id),
            device=self.device,
            max_new_tokens=self.generation_max_new_tokens,
            stop_sequences=stop_sequences,
            max_prompt_tokens=max_prompt_tokens,
        )
        self.agents = {self.agent_a.node_id: self.agent_a, self.agent_b.node_id: self.agent_b}

    @classmethod
    def from_checkpoint(cls, config: AgentRunnerConfig) -> "AgentRunner":
        if not config.checkpoint_dir_path:
            raise ValueError("checkpoint_dir_path is required")
        train_config_path = get_train_config_path(config.checkpoint_dir_path)
        if not train_config_path.exists():
            raise FileNotFoundError(f"Train config not found: {train_config_path}")
        train_payload = read_json(train_config_path)
        nodes, edges = build_nodes_and_edges(train_payload["model_ids"], train_payload["model_directions"])
        train_mod = importlib.import_module(f"{config.alg}.train")
        loaded = train_mod.load_translator_pool_from_checkpoint(
            checkpoint_dir_path=config.checkpoint_dir_path,
            nodes=nodes,
            edges=edges,
            device_override=resolve_device(config.device),
        )
        ctx, translator_pool, *_ = loaded
        set_seed(config.seed)
        return cls(
            ctx=ctx,
            translator_pool=translator_pool,
            alg=config.alg,
            max_turns=config.max_turns,
            generation_max_new_tokens=config.generation_max_new_tokens,
            max_prompt_tokens=config.max_prompt_tokens,
            seed=config.seed,
            log_turns=config.log_turns,
            log_max_chars=config.log_max_chars,
            cache_mode=config.cache_mode,
        )

    @staticmethod
    def build_initial_prompt(context: str, question: str) -> str:
        return (
            "You are Agent A, the hub agent in a two-agent QA discussion.\n"
            "Use the passage to answer the question briefly. If uncertain, propose a candidate answer for Agent B to verify.\n\n"
            f"Passage:\n{context.strip()}\n\n"
            f"Question: {question.strip()}\n"
            "Agent A:"
        )

    @staticmethod
    def build_followup_prompt(agent_id: str, turn_index: int) -> str:
        if agent_id == "B":
            return (
                "\nAgent B: Review the hub agent's answer against the passage. "
                "Either improve it or, when the answer is clear, start your reply with FINAL followed by a colon and a short answer.\n"
                "Agent B:"
            )
        return (
            "\nAgent A: Incorporate Agent B's reply. "
            "When enough evidence has been exchanged, start your reply with FINAL followed by a colon and a short answer; otherwise continue briefly.\n"
            "Agent A:"
        )

    @staticmethod
    def _append_turn_to_transcript(transcript: str, agent_id: str, response: str) -> str:
        clean_response = response.strip() or "[empty]"
        return f"{transcript}{clean_response}\n"

    @staticmethod
    def extract_final_answer(transcript: str, fallback_response: str) -> str:
        matches = list(re.finditer(r"FINAL\s*:\s*(.+)", transcript, flags=re.IGNORECASE))
        if matches:
            candidate = matches[-1].group(1).strip()
            candidate = re.split(r"[\n\r]", candidate, maxsplit=1)[0].strip()
            return postprocess_generated_answer(candidate)
        return postprocess_generated_answer(fallback_response)

    @staticmethod
    def _preview_text(text: str, max_chars: int) -> str:
        clean = re.sub(r"\s+", " ", (text or "").strip())
        if len(clean) <= max_chars:
            return clean
        return clean[: max(0, max_chars - 3)] + "..."

    def _log_example_start(
        self,
        *,
        question: str,
        gold_answers: Sequence[str],
        example_index: Optional[int],
    ) -> None:
        if not self.log_turns:
            return
        label = "?" if example_index is None else str(example_index)
        logging.info(
            "[AgentRunner][example=%s] start | mode=%s | alg=%s | question=%s | gold=%s",
            label,
            self.cache_mode,
            self.alg,
            self._preview_text(question, self.log_max_chars),
            list(gold_answers)[:3],
        )

    def _log_turn(self, *, example_index: Optional[int], turn_index: int, record: AgentTurnRecord) -> None:
        if not self.log_turns:
            return
        label = "?" if example_index is None else str(example_index)
        direction = "native-prefill" if record.translated_from is None else f"{record.translated_from}->{record.agent_id}"
        if record.cache_mode == CACHE_MODE_FREE and record.agent_id == self.agent_a.node_id and record.translated_from is None:
            direction = "hub-retained"
        logging.info(
            "[AgentRunner][example=%s][turn=%d] mode=%s | agent=%s | source=%s | stop=%s | "
            "cache=%d->%d | translated_delta_tokens=%d | cleared=%s",
            label,
            turn_index,
            record.cache_mode,
            record.agent_id,
            direction,
            record.stop_reason,
            record.cache_seq_len_before,
            record.cache_seq_len_after,
            record.translated_delta_tokens,
            record.cache_cleared_after_turn,
        )
        if record.translated_from is not None:
            logging.info(
                "[AgentRunner][example=%s][turn=%d] replay edge=%s | source_seq_len=%d | target_seq_len=%d",
                label,
                turn_index,
                record.translated_edge_id,
                record.translated_source_seq_len,
                record.translated_target_seq_len,
            )
        if record.offloaded_to is not None:
            logging.info(
                "[AgentRunner][example=%s][turn=%d] offload edge=%s | %s->%s | source_seq_len=%d | target_seq_len=%d | delta_tokens=%d",
                label,
                turn_index,
                record.offload_edge_id,
                record.agent_id,
                record.offloaded_to,
                record.offload_source_seq_len,
                record.offload_target_seq_len,
                record.offload_delta_tokens,
            )
        logging.info(
            "[AgentRunner][example=%s][turn=%d] prompt: %s",
            label,
            turn_index,
            self._preview_text(record.prompt, self.log_max_chars),
        )
        logging.info(
            "[AgentRunner][example=%s][turn=%d] response: %s",
            label,
            turn_index,
            self._preview_text(record.response or record.raw_response, self.log_max_chars),
        )

    def _log_example_end(self, *, example_index: Optional[int], prediction: str, f1: float) -> None:
        if not self.log_turns:
            return
        label = "?" if example_index is None else str(example_index)
        logging.info(
            "[AgentRunner][example=%s] end | mode=%s | prediction=%s | f1=%.4f",
            label,
            self.cache_mode,
            self._preview_text(prediction, self.log_max_chars),
            f1,
        )

    def _translate_into(self, *, source_agent: Agent, target_agent: Agent, transcript: str) -> Dict[str, Any]:
        translated_past, metadata = self.cache_translator.translate_full_cache(
            source_agent=source_agent,
            target_agent=target_agent,
            transcript_text=transcript,
        )
        target_agent.set_replayed_cache(translated_past, transcript_text=transcript)
        return metadata

    def _clear_non_hub_cache(self, agent: Agent) -> bool:
        if agent.node_id == self.agent_a.node_id:
            return False
        agent.clear_kv_cache(empty_cuda_cache=True)
        return True

    def _run_retain_turns(
        self,
        *,
        transcript: str,
        initial_generation: AgentGeneration,
        turns: List[AgentTurnRecord],
        example_index: Optional[int],
    ) -> Tuple[str, str]:
        last_response = initial_generation.text
        current_source = self.agent_a
        current_target = self.agent_b
        for turn_index in range(1, max(1, self.max_turns)):
            translation_meta = self._translate_into(
                source_agent=current_source,
                target_agent=current_target,
                transcript=transcript,
            )
            prompt = self.build_followup_prompt(current_target.node_id, turn_index)
            generation = current_target.generate_response(prompt)
            transcript = self._append_turn_to_transcript(transcript + prompt, current_target.node_id, generation.text)
            record = self._turn_record(
                generation,
                translated_from=current_source.node_id,
                translated_edge_id=str(translation_meta.get("edge_id", "")) or None,
                translated_source_seq_len=int(translation_meta.get("source_seq_len", 0)),
                translated_target_seq_len=int(translation_meta.get("target_seq_len", 0)),
                translated_delta_tokens=int(translation_meta.get("delta_tokens", 0)),
            )
            turns.append(record)
            self._log_turn(example_index=example_index, turn_index=turn_index, record=record)
            last_response = generation.text
            if "FINAL" in generation.text.upper():
                break
            current_source, current_target = current_target, current_source
        return transcript, last_response

    def _run_free_turns(
        self,
        *,
        transcript: str,
        initial_generation: AgentGeneration,
        turns: List[AgentTurnRecord],
        example_index: Optional[int],
    ) -> Tuple[str, str]:
        last_response = initial_generation.text
        for turn_index in range(1, max(1, self.max_turns)):
            if turn_index % 2 == 1:
                # Non-hub Agent B owns no cache between turns in free mode.
                # It receives the full Hub A cache, replays from scratch, generates,
                # offloads the newly extended state back to A, and is then cleared.
                translation_meta = self._translate_into(
                    source_agent=self.agent_a,
                    target_agent=self.agent_b,
                    transcript=transcript,
                )
                prompt = self.build_followup_prompt(self.agent_b.node_id, turn_index)
                generation = self.agent_b.generate_response(prompt)
                transcript = self._append_turn_to_transcript(transcript + prompt, self.agent_b.node_id, generation.text)
                offload_meta = self._translate_into(
                    source_agent=self.agent_b,
                    target_agent=self.agent_a,
                    transcript=transcript,
                )
                cleared = self._clear_non_hub_cache(self.agent_b)
                record = self._turn_record(
                    generation,
                    translated_from=self.agent_a.node_id,
                    translated_edge_id=str(translation_meta.get("edge_id", "")) or None,
                    translated_source_seq_len=int(translation_meta.get("source_seq_len", 0)),
                    translated_target_seq_len=int(translation_meta.get("target_seq_len", 0)),
                    translated_delta_tokens=int(translation_meta.get("delta_tokens", 0)),
                    offloaded_to=self.agent_a.node_id,
                    offload_edge_id=str(offload_meta.get("edge_id", "")) or None,
                    offload_source_seq_len=int(offload_meta.get("source_seq_len", 0)),
                    offload_target_seq_len=int(offload_meta.get("target_seq_len", 0)),
                    offload_delta_tokens=int(offload_meta.get("delta_tokens", 0)),
                    cache_cleared_after_turn=cleared,
                )
            else:
                # Hub Agent A retains the canonical full conversation cache.
                prompt = self.build_followup_prompt(self.agent_a.node_id, turn_index)
                generation = self.agent_a.generate_response(prompt)
                transcript = self._append_turn_to_transcript(transcript + prompt, self.agent_a.node_id, generation.text)
                record = self._turn_record(generation)

            turns.append(record)
            self._log_turn(example_index=example_index, turn_index=turn_index, record=record)
            last_response = generation.text
            if "FINAL" in generation.text.upper():
                break
        return transcript, last_response

    def _run_example_impl(
        self,
        *,
        context: str,
        question: str,
        gold_answers: Sequence[str],
        example_index: Optional[int] = None,
    ) -> AgentRunnerResult:
        self.agent_a.reset()
        self.agent_b.reset()
        self.cache_translator._sent_source_seq_lens.clear()
        self.translator_pool.eval()
        for node in self.ctx.nodes:
            self.ctx.mm.get_model(node.id).eval()

        self._log_example_start(question=question, gold_answers=gold_answers, example_index=example_index)
        transcript = self.build_initial_prompt(context, question)
        turns: List[AgentTurnRecord] = []

        # Agent A starts from native Base Context + Prompt and acts as the hub.
        generation = self.agent_a.generate_response(transcript)
        transcript = self._append_turn_to_transcript(transcript, self.agent_a.node_id, generation.text)
        record = self._turn_record(generation)
        turns.append(record)
        self._log_turn(example_index=example_index, turn_index=0, record=record)

        last_response = generation.text
        if "FINAL" not in generation.text.upper():
            if self.cache_mode == CACHE_MODE_FREE:
                transcript, last_response = self._run_free_turns(
                    transcript=transcript,
                    initial_generation=generation,
                    turns=turns,
                    example_index=example_index,
                )
            else:
                transcript, last_response = self._run_retain_turns(
                    transcript=transcript,
                    initial_generation=generation,
                    turns=turns,
                    example_index=example_index,
                )

        prediction = self.extract_final_answer(transcript, last_response)
        f1 = compute_generation_f1(prediction, list(gold_answers))
        self._log_example_end(example_index=example_index, prediction=prediction, f1=f1)
        return AgentRunnerResult(
            question=question,
            gold_answers=list(gold_answers),
            prediction=prediction,
            f1=f1,
            transcript=transcript,
            turns=turns,
            profile={},
        )

    def _turn_record(
        self,
        generation: AgentGeneration,
        *,
        translated_from: Optional[str] = None,
        translated_edge_id: Optional[str] = None,
        translated_source_seq_len: int = 0,
        translated_target_seq_len: int = 0,
        translated_delta_tokens: int = 0,
        offloaded_to: Optional[str] = None,
        offload_edge_id: Optional[str] = None,
        offload_source_seq_len: int = 0,
        offload_target_seq_len: int = 0,
        offload_delta_tokens: int = 0,
        cache_cleared_after_turn: bool = False,
    ) -> AgentTurnRecord:
        return AgentTurnRecord(
            agent_id=generation.agent_id,
            prompt=generation.prompt_text,
            response=generation.text,
            raw_response=generation.raw_text,
            stop_reason=generation.stop_reason,
            cache_seq_len_before=generation.cache_seq_len_before,
            cache_seq_len_after=generation.cache_seq_len_after,
            cache_mode=self.cache_mode,
            translated_from=translated_from,
            translated_edge_id=translated_edge_id,
            translated_source_seq_len=translated_source_seq_len,
            translated_target_seq_len=translated_target_seq_len,
            translated_delta_tokens=translated_delta_tokens,
            offloaded_to=offloaded_to,
            offload_edge_id=offload_edge_id,
            offload_source_seq_len=offload_source_seq_len,
            offload_target_seq_len=offload_target_seq_len,
            offload_delta_tokens=offload_delta_tokens,
            cache_cleared_after_turn=cache_cleared_after_turn,
        )

    def run(
        self,
        *,
        context: str,
        question: str,
        gold_answers: Sequence[str],
        example_index: Optional[int] = None,
    ) -> AgentRunnerResult:
        token_budget = max(1, self.max_turns) * max(1, self.generation_max_new_tokens)
        result, profile = self.profiler.measure(
            lambda: self._run_example_impl(
                context=context,
                question=question,
                gold_answers=gold_answers,
                example_index=example_index,
            ),
            tokens=token_budget,
        )
        result.profile = profile
        return result


__all__ = [
    "CACHE_MODE_FREE",
    "CACHE_MODE_RETAIN",
    "SUPPORTED_CACHE_MODES",
    "AgentRunner",
    "AgentRunnerConfig",
    "AgentRunnerResult",
    "AgentTurnRecord",
    "KVCacheTranslationAdapter",
    "get_squad_v11_dataset_spec",
]
