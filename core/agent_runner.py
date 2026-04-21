from __future__ import annotations

import copy
import gc
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .agent import Agent, GenerationOutput, HubAgent, sum_live_kv_bytes
from .common import PastKeyValues, build_timestamp_string, write_json
from .context import Context
from .dialogue_dataset import DialogueEpisode
from .eval_util import EvalConfig, HFDatasetSpec, compute_benchmark_context_budget, compute_generation_f1


SHARED_PROMPT_HEADER = (
    "You are participating in a document-grounded multi-turn question answering task.\n"
    "Use only the information in [DOCUMENT] and [HISTORY].\n"
    "Do not use outside knowledge.\n"
    "If the excerpt does not contain enough information, say so briefly.\n"
    "Answer clearly, concisely, and in a grounded way."
)


@dataclass
class TurnRunResult:
    qa_turn_idx: int
    user_turn_id: int
    agent_turn_id: int
    user_utterance: str
    gold_answer: str
    a_raw_response: str
    b_raw_response: str
    final_response: str
    f1: float
    stage_token_counts: Dict[str, int]
    context_budgets: Dict[str, int]


@dataclass
class EpisodeRunResult:
    policy: str
    dial_id: str
    domain: str
    doc_id: str
    title: str
    stop_reason: Optional[str]
    turns_attempted: int
    turns_completed: int
    average_f1: float
    peak_live_kv_bytes: int
    turn_results: List[TurnRunResult]


class AgentRunner:
    def __init__(
        self,
        *,
        ctx: Context,
        translator_pool,
        eval_config: EvalConfig,
        policy: str,
        draft_max_new_tokens: int = 48,
        max_turns_per_episode: int = 10,
        debug_enabled: bool = False,
        debug_output_path: Optional[str] = None,
    ) -> None:
        normalized_policy = str(policy).strip().lower()
        if normalized_policy not in {"retain", "free"}:
            raise ValueError("policy must be one of {'retain', 'free'}")

        self.ctx = ctx
        self.translator_pool = translator_pool
        self.eval_config = eval_config
        self.policy = normalized_policy
        self.draft_max_new_tokens = max(1, int(draft_max_new_tokens))
        self.max_turns_per_episode = max(1, int(max_turns_per_episode))
        self.debug_enabled = debug_enabled
        self.debug_output_path = Path(debug_output_path) if debug_output_path else None

        self.node_ids = [node.id for node in ctx.nodes]
        if self.node_ids != ["A", "B"]:
            raise ValueError(
                "This runner expects exactly two nodes named A and B. "
                f"Got nodes={self.node_ids}"
            )

        self.edge_ids = {edge.id for edge in ctx.edges}
        required_edges = {"A_to_B", "B_to_A"}
        if not required_edges.issubset(self.edge_ids):
            raise ValueError(
                "This runner expects bidirectional edges A_to_B and B_to_A. "
                f"Got edges={sorted(self.edge_ids)}"
            )

        self.a_agent = HubAgent(
            node_id="A",
            model=ctx.mm.get_model("A"),
            tokenizer=ctx.tokenizer,
            device=ctx.config.device,
        )
        self.b_agent = Agent(
            node_id="B",
            model=ctx.mm.get_model("B"),
            tokenizer=ctx.tokenizer,
            device=ctx.config.device,
        )

        self.generation_spec = HFDatasetSpec(
            name_for_log="doc2dial_multiturn",
            dataset_path="doc2dial",
            dataset_name="dialogue_domain",
            split="validation",
            answer_mode="squad",
            question_field="question",
            context_field="context",
            answers_field="answers",
            streaming=False,
        )

        if self.debug_output_path is not None:
            self.debug_output_path.parent.mkdir(parents=True, exist_ok=True)

    def run_episode(self, episode: DialogueEpisode) -> EpisodeRunResult:
        self.translator_pool.eval()
        base_text = self.build_base_text(episode)
        boundary_token_ids = self.a_agent.encode_text(base_text)
        if len(boundary_token_ids) < 1:
            raise ValueError(f"Episode {episode.dial_id} produced an empty base transcript.")

        boundary_past_a = self.a_agent.build_past_from_token_ids(boundary_token_ids)
        boundary_past_b: Optional[PastKeyValues] = None
        if self.policy == "retain":
            boundary_past_b = self.b_agent.build_past_from_token_ids(boundary_token_ids)

        live_past_map: Dict[str, Optional[PastKeyValues]] = {
            "A_boundary": boundary_past_a,
            "B_boundary": boundary_past_b,
        }
        peak_live_kv_bytes = self._update_peak_live_kv(live_past_map, current_peak=0)

        turn_results: List[TurnRunResult] = []
        stop_reason: Optional[str] = None
        turns_attempted = 0

        for qa_turn in episode.qa_turns[: self.max_turns_per_episode]:
            turns_attempted += 1
            user_line = self.build_user_line(qa_turn.qa_turn_idx, qa_turn.user_utterance)
            user_line_ids = self.a_agent.encode_text(user_line)

            a_prompt_ids = self.a_agent.encode_text(self.build_stage_prompt("A_DRAFT_t"))
            if len(a_prompt_ids) < 1:
                raise ValueError("A_DRAFT_t prompt must tokenize to at least one token.")
            a_prefix_ids = boundary_token_ids + user_line_ids + a_prompt_ids
            a_budget = self._compute_context_budget(
                question=qa_turn.user_utterance,
                generation_max_new_tokens=self.draft_max_new_tokens,
            )
            if len(a_prefix_ids) > a_budget:
                stop_reason = f"context_budget_exceeded_before_A@turn_{qa_turn.qa_turn_idx}"
                break

            a_cache_append_ids = user_line_ids + a_prompt_ids[:-1]
            a_cache_past = self.a_agent.append_token_ids_to_past(boundary_past_a, a_cache_append_ids)
            live_past_map["A_turn_cache"] = a_cache_past
            peak_live_kv_bytes = self._update_peak_live_kv(live_past_map, current_peak=peak_live_kv_bytes)

            a_output = self.a_agent.generate_from_past_and_seed(
                past_key_values=a_cache_past,
                seed_token_id=a_prompt_ids[-1],
                max_new_tokens=self.draft_max_new_tokens,
            )
            live_past_map["A_turn_cache"] = None
            live_past_map["A_after_draft"] = a_output.final_past_key_values
            peak_live_kv_bytes = self._update_peak_live_kv(live_past_map, current_peak=peak_live_kv_bytes)
            if self.debug_enabled:
                self.a_agent.debug(f"{episode.dial_id}/turn_{qa_turn.qa_turn_idx}/A_DRAFT", a_output)

            b_prompt_ids = self.a_agent.encode_text(self.build_stage_prompt("B_DRAFT_t"))
            if len(b_prompt_ids) < 1:
                raise ValueError("B_DRAFT_t prompt must tokenize to at least one token.")
            a_consumed_ids = a_prefix_ids + a_output.generated_token_ids
            b_prefix_ids = a_consumed_ids + b_prompt_ids
            b_budget = self._compute_context_budget(
                question=qa_turn.user_utterance,
                generation_max_new_tokens=self.draft_max_new_tokens,
            )
            if len(b_prefix_ids) > b_budget:
                stop_reason = f"context_budget_exceeded_before_B@turn_{qa_turn.qa_turn_idx}"
                break

            source_past_for_b: Optional[PastKeyValues] = None
            reused_b_boundary = False
            b_delta_ids = user_line_ids + a_prompt_ids + a_output.generated_token_ids + b_prompt_ids[:-1]
            if self.policy == "retain" and boundary_past_b is not None:
                mixed_b_past = self.b_agent.append_token_ids_to_past(boundary_past_b, b_delta_ids)
                reused_b_boundary = True
            else:
                source_past_for_b = self.a_agent.append_token_ids_to_past(
                    a_output.final_past_key_values,
                    b_prompt_ids[:-1],
                )
                live_past_map["A_source_for_B"] = source_past_for_b
                peak_live_kv_bytes = self._update_peak_live_kv(live_past_map, current_peak=peak_live_kv_bytes)

                mixed_b_past = self.offload_past(
                    source_past_key_values=source_past_for_b,
                    prefix_cache_token_ids=b_prefix_ids[:-1],
                    src_node_id="A",
                    tgt_node_id="B",
                )
                live_past_map["A_source_for_B"] = None

            live_past_map["B_translated"] = mixed_b_past
            peak_live_kv_bytes = self._update_peak_live_kv(live_past_map, current_peak=peak_live_kv_bytes)

            b_output = self.b_agent.generate_from_past_and_seed(
                past_key_values=mixed_b_past,
                seed_token_id=b_prompt_ids[-1],
                max_new_tokens=self.draft_max_new_tokens,
            )
            live_past_map["B_translated"] = None
            live_past_map["B_after_draft"] = b_output.final_past_key_values
            peak_live_kv_bytes = self._update_peak_live_kv(live_past_map, current_peak=peak_live_kv_bytes)
            if self.debug_enabled:
                self.b_agent.debug(f"{episode.dial_id}/turn_{qa_turn.qa_turn_idx}/B_DRAFT", b_output)

            final_prompt_ids = self.a_agent.encode_text(self.build_stage_prompt("FINAL_t"))
            if len(final_prompt_ids) < 1:
                raise ValueError("FINAL_t prompt must tokenize to at least one token.")
            b_consumed_ids = b_prefix_ids + b_output.generated_token_ids
            final_prefix_ids = b_consumed_ids + final_prompt_ids
            final_budget = self._compute_context_budget(
                question=qa_turn.user_utterance,
                generation_max_new_tokens=self.eval_config.generation_max_new_tokens,
            )
            if len(final_prefix_ids) > final_budget:
                stop_reason = f"context_budget_exceeded_before_FINAL@turn_{qa_turn.qa_turn_idx}"
                break

            source_past_for_final = self.b_agent.append_token_ids_to_past(
                b_output.final_past_key_values,
                final_prompt_ids[:-1],
            )
            live_past_map["B_source_for_A"] = source_past_for_final
            peak_live_kv_bytes = self._update_peak_live_kv(live_past_map, current_peak=peak_live_kv_bytes)

            mixed_a_final_past = self.offload_past(
                source_past_key_values=source_past_for_final,
                prefix_cache_token_ids=final_prefix_ids[:-1],
                src_node_id="B",
                tgt_node_id="A",
            )
            live_past_map["B_source_for_A"] = None
            live_past_map["A_final_translated"] = mixed_a_final_past
            peak_live_kv_bytes = self._update_peak_live_kv(live_past_map, current_peak=peak_live_kv_bytes)

            final_output = self.a_agent.generate_from_past_and_seed(
                past_key_values=mixed_a_final_past,
                seed_token_id=final_prompt_ids[-1],
                max_new_tokens=self.eval_config.generation_max_new_tokens,
            )
            live_past_map["A_final_translated"] = None
            if self.debug_enabled:
                self.a_agent.debug(f"{episode.dial_id}/turn_{qa_turn.qa_turn_idx}/FINAL", final_output)

            f1 = compute_generation_f1(final_output.text, [qa_turn.gold_answer])
            stage_token_counts = {
                "A_prefix_tokens": len(a_prefix_ids),
                "A_generated_tokens": a_output.generated_tokens,
                "B_prefix_tokens": len(b_prefix_ids),
                "B_generated_tokens": b_output.generated_tokens,
                "FINAL_prefix_tokens": len(final_prefix_ids),
                "FINAL_generated_tokens": final_output.generated_tokens,
                "B_used_boundary_cache": int(reused_b_boundary),
                "B_delta_tokens": len(b_delta_ids),
            }
            context_budgets = {
                "A_budget": a_budget,
                "B_budget": b_budget,
                "FINAL_budget": final_budget,
            }
            turn_result = TurnRunResult(
                qa_turn_idx=qa_turn.qa_turn_idx,
                user_turn_id=qa_turn.user_turn_id,
                agent_turn_id=qa_turn.agent_turn_id,
                user_utterance=qa_turn.user_utterance,
                gold_answer=qa_turn.gold_answer,
                a_raw_response=a_output.text,
                b_raw_response=b_output.text,
                final_response=final_output.text,
                f1=f1,
                stage_token_counts=stage_token_counts,
                context_budgets=context_budgets,
            )
            turn_results.append(turn_result)
            self.debug_turn(episode=episode, turn_result=turn_result)

            clean_history_update = self.build_history_update(
                qa_turn_idx=qa_turn.qa_turn_idx,
                user_utterance=qa_turn.user_utterance,
                final_response=final_output.text,
            )
            clean_history_ids = self.a_agent.encode_text(clean_history_update)
            boundary_past_a = self.a_agent.append_token_ids_to_past(boundary_past_a, clean_history_ids)
            if self.policy == "retain":
                if boundary_past_b is None:
                    boundary_past_b = self.b_agent.build_past_from_token_ids(boundary_token_ids)
                boundary_past_b = self.b_agent.append_token_ids_to_past(boundary_past_b, clean_history_ids)
            else:
                boundary_past_b = None
            boundary_token_ids = boundary_token_ids + clean_history_ids
            live_past_map["A_boundary"] = boundary_past_a
            live_past_map["B_boundary"] = boundary_past_b
            peak_live_kv_bytes = self._update_peak_live_kv(live_past_map, current_peak=peak_live_kv_bytes)

            live_past_map["A_after_draft"] = None
            live_past_map["B_after_draft"] = None
            if self.policy == "free":
                self.free_past(b_output.final_past_key_values)
                self.free_past(source_past_for_final)
                self.free_past(mixed_b_past)
            elif source_past_for_b is not None:
                self.free_past(source_past_for_b)

        average_f1 = sum(turn.f1 for turn in turn_results) / len(turn_results) if turn_results else 0.0
        return EpisodeRunResult(
            policy=self.policy,
            dial_id=episode.dial_id,
            domain=episode.domain,
            doc_id=episode.doc_id,
            title=episode.title,
            stop_reason=stop_reason,
            turns_attempted=turns_attempted,
            turns_completed=len(turn_results),
            average_f1=average_f1,
            peak_live_kv_bytes=int(peak_live_kv_bytes),
            turn_results=turn_results,
        )

    def build_base_text(self, episode: DialogueEpisode) -> str:
        grounding = (episode.gold_grounding_excerpt or "").strip()
        title = (episode.title or "").strip()
        return (
            "[TASK]\n"
            f"{SHARED_PROMPT_HEADER}\n\n"
            "[DOCUMENT]\n"
            f"{grounding}\n\n"
            "[TITLE]\n"
            f"{title}\n\n"
            "[CURRENT QUESTION POLICY]\n"
            "Answer the current user question using the document excerpt and dialogue history.\n"
            "Prefer exact details from the excerpt. Do not invent unsupported facts.\n\n"
            "[HISTORY]\n"
        )

    def build_user_line(self, qa_turn_idx: int, user_utterance: str) -> str:
        return f"User_{qa_turn_idx}: {user_utterance.strip()}\n"

    def build_stage_prompt(self, stage_name: str) -> str:
        prompts = {
            "A_DRAFT_t": (
                "A_DRAFT_t: Draft a short grounded answer for the current user question. "
                "Focus on the most relevant facts from the document excerpt.\nAnswer:"
            ),
            "B_DRAFT_t": (
                "\nB_DRAFT_t: Improve the draft for accuracy, completeness, and clarity. "
                "Keep it grounded in the document excerpt and history.\nImproved answer:"
            ),
            "FINAL_t": (
                "\nFINAL_t: Write the final answer to the user. "
                "Use the excerpt, history, and the two drafts. "
                "Be concise and directly answer the question.\nFinal answer:"
            ),
        }
        if stage_name not in prompts:
            raise ValueError(f"Unknown stage name: {stage_name}")
        return prompts[stage_name]

    def build_history_update(self, *, qa_turn_idx: int, user_utterance: str, final_response: str) -> str:
        return (
            f"User_{qa_turn_idx}: {user_utterance.strip()}\n"
            f"Final_{qa_turn_idx}: {final_response.strip()}\n"
        )

    def offload_past(
        self,
        *,
        source_past_key_values: PastKeyValues,
        prefix_cache_token_ids: List[int],
        src_node_id: str,
        tgt_node_id: str,
    ) -> PastKeyValues:
        prefix_cache_tensor = self.a_agent.to_tensor(prefix_cache_token_ids)
        mixed_target_past, _ = self.translator_pool.build_replayed_target_past(
            source_past_key_values=source_past_key_values,
            prefix_input_ids=prefix_cache_tensor,
            target_model=self.ctx.mm.get_model(tgt_node_id),
            src_node_id=src_node_id,
            tgt_node_id=tgt_node_id,
            tgt_spec=self.ctx.mm.get_model_spec(tgt_node_id),
        )
        return mixed_target_past

    def free_past(self, past_key_values: Optional[PastKeyValues]) -> None:
        del past_key_values
        gc.collect()
        if torch.cuda.is_available() and str(self.ctx.config.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def debug_turn(self, *, episode: DialogueEpisode, turn_result: TurnRunResult) -> None:
        payload = {
            "policy": self.policy,
            "dial_id": episode.dial_id,
            "domain": episode.domain,
            "doc_id": episode.doc_id,
            "title": episode.title,
            **asdict(turn_result),
        }
        if self.debug_enabled:
            print(
                f"[{self.policy}] {episode.dial_id} turn={turn_result.qa_turn_idx} | "
                f"A_RAW={turn_result.a_raw_response!r} | "
                f"B_RAW={turn_result.b_raw_response!r} | "
                f"FINAL={turn_result.final_response!r} | "
                f"F1={turn_result.f1:.4f}"
            )
        if self.debug_output_path is not None:
            with self.debug_output_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _compute_context_budget(self, *, question: str, generation_max_new_tokens: int) -> int:
        stage_eval_config = copy.copy(self.eval_config)
        stage_eval_config.generation_max_new_tokens = int(generation_max_new_tokens)
        return compute_benchmark_context_budget(
            ctx=self.ctx,
            spec=self.generation_spec,
            question=question,
            eval_config=stage_eval_config,
        )

    @staticmethod
    def _update_peak_live_kv(live_past_map: Dict[str, Optional[PastKeyValues]], current_peak: int) -> int:
        live_bytes = sum_live_kv_bytes(live_past_map)
        return max(int(current_peak), int(live_bytes))


@dataclass
class RunnerSummary:
    policy: str
    num_episodes: int
    num_completed_episodes: int
    num_turns_completed: int
    average_episode_f1: float
    average_turn_f1: float
    average_peak_live_kv_mib: float
    max_peak_live_kv_mib: float
    stop_reason_counts: Dict[str, int]
    results_json_path: str
    debug_jsonl_path: Optional[str]


def run_agent_policy(
    *,
    runner: AgentRunner,
    episodes: List[DialogueEpisode],
    output_dir: str,
) -> RunnerSummary:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    episode_results: List[Dict[str, object]] = []
    stop_reasons = Counter()
    total_turn_f1 = 0.0
    total_turn_count = 0
    total_episode_f1 = 0.0
    peak_values_mib: List[float] = []
    completed_episodes = 0

    for episode in episodes:
        result = runner.run_episode(episode)
        if result.stop_reason:
            stop_reasons[result.stop_reason] += 1
        if result.turn_results:
            completed_episodes += 1
        total_episode_f1 += result.average_f1
        total_turn_count += len(result.turn_results)
        total_turn_f1 += sum(turn.f1 for turn in result.turn_results)
        peak_values_mib.append(result.peak_live_kv_bytes / (1024 ** 2))
        episode_results.append(
            {
                "policy": result.policy,
                "dial_id": result.dial_id,
                "domain": result.domain,
                "doc_id": result.doc_id,
                "title": result.title,
                "stop_reason": result.stop_reason,
                "turns_attempted": result.turns_attempted,
                "turns_completed": result.turns_completed,
                "average_f1": result.average_f1,
                "peak_live_kv_bytes": result.peak_live_kv_bytes,
                "peak_live_kv_mib": result.peak_live_kv_bytes / (1024 ** 2),
                "turn_results": [asdict(turn) for turn in result.turn_results],
            }
        )

    results_json_path = output_dir_path / f"{runner.policy}_episode_results.json"
    write_json(str(results_json_path), episode_results)

    average_episode_f1 = total_episode_f1 / len(episodes) if episodes else 0.0
    average_turn_f1 = total_turn_f1 / total_turn_count if total_turn_count > 0 else 0.0
    average_peak_live_kv_mib = sum(peak_values_mib) / len(peak_values_mib) if peak_values_mib else 0.0
    max_peak_live_kv_mib = max(peak_values_mib) if peak_values_mib else 0.0

    return RunnerSummary(
        policy=runner.policy,
        num_episodes=len(episodes),
        num_completed_episodes=completed_episodes,
        num_turns_completed=total_turn_count,
        average_episode_f1=average_episode_f1,
        average_turn_f1=average_turn_f1,
        average_peak_live_kv_mib=average_peak_live_kv_mib,
        max_peak_live_kv_mib=max_peak_live_kv_mib,
        stop_reason_counts=dict(stop_reasons),
        results_json_path=str(results_json_path),
        debug_jsonl_path=str(runner.debug_output_path) if runner.debug_output_path is not None else None,
    )


def build_default_multiturn_output_dir(base_dir: str) -> str:
    timestamp = build_timestamp_string()
    return str(Path(base_dir) / f"multiturn_qa_{timestamp}")
