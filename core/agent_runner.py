from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from core.agent import Agent, AgentGeneration, HubAgent, get_past_seq_len, slice_past_suffix
from core.common import PastKeyValues, TokenIDs, extract_past_key_values, read_json, replace_top_layers, set_seed
from core.config import resolve_device
from core.context import Context
from core.eval_util import (
    GPUMemoryBreakdownBytes,
    measure_gpu_memory_breakdown_bytes,
    postprocess_generated_answer,
)
from core.topology import Edge, Node, build_edge_map, build_nodes_and_edges, index_to_node_id
from core.train_util import get_train_config_path


CACHE_MODE_RETAIN = "retain"
CACHE_MODE_FREE = "free"
SUPPORTED_CACHE_MODES = (CACHE_MODE_RETAIN, CACHE_MODE_FREE)

SUPPORTED_ALGS = ("mot", "interlat", "lsc", "c2c-pr", "kvcomm")
RETAIN_ONLY_ALGS = ("interlat", "lsc", "c2c-pr", "kvcomm")
TRAIN_MODULE_BY_ALG = {
    "mot": "alg.mot.train",
    "interlat": "alg.interlat.train",
    "lsc": "alg.lsc.train",
    "c2c-pr": "alg.c2c.train",
    "kvcomm": "alg.kvcomm.train",
}


def _is_homogeneous_model_pool(nodes: Sequence[Node]) -> bool:
    return len({node.model_id for node in nodes}) == 1


def normalize_agent_runner_alg(alg: str) -> str:
    normalized = str(alg).strip().lower()
    if normalized not in SUPPORTED_ALGS:
        raise ValueError(f"Unsupported alg={alg!r}; expected one of {SUPPORTED_ALGS}")
    return normalized


def resolve_agent_count(
    agent_count: Optional[int],
    total_nodes: int,
) -> int:
    total_nodes = int(total_nodes)
    if total_nodes < 2:
        raise ValueError("AgentRunner requires at least two nodes in the checkpoint model pool.")
    if agent_count is None:
        return total_nodes
    resolved = int(agent_count)
    if resolved < 1:
        raise ValueError(f"agent_count must be at least 1, got {resolved}")
    return resolved


OFFLOAD_KIND_DELTA = "delta"


def _concat_past_key_values(prefix: Optional[PastKeyValues], suffix: PastKeyValues) -> PastKeyValues:
    """Append a translated suffix cache to an existing target-side cache."""
    if prefix is None:
        return suffix
    if len(prefix) != len(suffix):
        raise ValueError(
            f"Cannot concatenate KV caches with different layer counts: "
            f"prefix={len(prefix)}, suffix={len(suffix)}"
        )
    concatenated = []
    for layer_idx, ((prefix_key, prefix_value), (suffix_key, suffix_value)) in enumerate(zip(prefix, suffix)):
        if prefix_key.shape[:2] != suffix_key.shape[:2] or prefix_key.shape[3:] != suffix_key.shape[3:]:
            raise ValueError(
                f"Cannot concatenate KV cache layer {layer_idx}: key shape mismatch "
                f"prefix={tuple(prefix_key.shape)}, suffix={tuple(suffix_key.shape)}"
            )
        if prefix_value.shape[:2] != suffix_value.shape[:2] or prefix_value.shape[3:] != suffix_value.shape[3:]:
            raise ValueError(
                f"Cannot concatenate KV cache layer {layer_idx}: value shape mismatch "
                f"prefix={tuple(prefix_value.shape)}, suffix={tuple(suffix_value.shape)}"
            )
        concatenated.append(
            (
                torch.cat([prefix_key, suffix_key], dim=2).contiguous(),
                torch.cat([prefix_value, suffix_value], dim=2).contiguous(),
            )
        )
    return tuple(concatenated)



@dataclass
class AgentRunnerConfig:
    alg: str
    checkpoint_dir_path: str = ""
    device: str = "auto"
    max_turns: int = 4
    generation_max_new_tokens: int = 48
    max_prompt_tokens: Optional[int] = None
    seed: int = 42
    log_turns: bool = True
    log_max_chars: int = 600
    cache_mode: str = CACHE_MODE_RETAIN
    agent_count: Optional[int] = None


@dataclass
class AgentTurnRecord:
    agent_id: str
    prompt: str
    response: str
    tokens_before: int
    tokens_after: int
    tokens_prompt: int
    tokens_completion: int
    cache_mode: str = CACHE_MODE_RETAIN
    is_hub: bool = False
    translated_edge_id: Optional[str] = None
    translated_offload_kind: Optional[str] = None
    tokens_received: int = 0
    offload_edge_id: Optional[str] = None
    offload_kind: Optional[str] = None
    tokens_sent: int = 0
    tokens_sent_check_passed: bool = True


@dataclass
class AgentRunnerResult:
    question: str
    gold_answers: List[str]
    prediction: str
    accuracy: float
    transcript: str
    turns: List[AgentTurnRecord]
    profile: Dict[str, Optional[float]]
    agent_ids: List[str]
    hub_agent_id: str
    cache_mode: str

    @property
    def peak_memory_gib(self) -> float:
        values = [
            self.profile.get("model_memory_gib"),
            self.profile.get("translator_memory_gib"),
            self.profile.get("kv_memory_gib"),
        ]
        if any(value is None for value in values):
            return float("nan")
        return sum(float(value) for value in values if value is not None)


class KVCacheTranslationAdapter:
    """Algorithm-aware helper for pretranslated KV-cache handoff.

    Delta sizing is token-id based. Translation is prepared before offload and
    stored on the source Agent, so offload_cache() only slices an already
    translated target-side cache piece.
    """

    def __init__(self, *, ctx: Context, translator_pool, alg: str) -> None:
        self.ctx = ctx
        self.translator_pool = translator_pool
        self.alg = normalize_agent_runner_alg(alg)
        self.edge_map = build_edge_map(ctx.edges)
        self._canonical_edge = next(iter(ctx.edges), None)

    def _get_edge(self, src_node_id: str, tgt_node_id: str) -> Edge:
        edge_id = f"{src_node_id}_to_{tgt_node_id}"
        edge = self.edge_map.get(edge_id)
        if edge is not None:
            return edge
        if self._canonical_edge is not None:
            # AgentRunner only accepts homogeneous checkpoints, so a single trained direction
            # such as A_to_B is the universal physical translator for every logical handoff.
            return self._canonical_edge
        raise ValueError(
            f"Missing translator edge {edge_id!r}. AgentRunner requires every KV offload edge used by the "
            f"selected hub-centered star topology/cache mode. Non-hub to non-hub handoffs are routed "
            f"through the hub and therefore require source_to_hub and hub_to_target translator edges. "
            f"Available edges: {sorted(self.edge_map)}"
        )

    @staticmethod
    def _build_token_ids(
        token_ids: Sequence[int],
        *,
        model_id: str,
        device: str,
    ) -> TokenIDs:
        ids = list(token_ids)
        if not ids:
            raise ValueError("Cannot prepare or offload an empty KV cache.")
        return TokenIDs(
            torch.tensor([ids], dtype=torch.long, device=device),
            model_id=model_id,
        )

    @torch.inference_mode()
    def build_pretranslated_past_for_edge(
        self,
        *,
        source_agent: Agent,
        target_agent: Agent,
        source_past_key_values: PastKeyValues,
        source_token_ids: Sequence[int],
    ) -> Tuple[str, PastKeyValues]:
        """Translate an already-built source cache into the target model space.

        This is used immediately after an Agent cache changes, never inside
        offload_cache(). The returned full target-side cache can later be sliced
        by token-id delta during the physical handoff.
        """
        token_ids = list(source_token_ids)
        source_tokens = get_past_seq_len(source_past_key_values)
        if len(token_ids) != source_tokens:
            raise ValueError(
                f"Source token-id ledger is not aligned with KV cache for "
                f"{source_agent.node_id}->{target_agent.node_id}: "
                f"token_ids={len(token_ids)} past_tokens={source_tokens}"
            )

        edge = self._get_edge(source_agent.node_id, target_agent.node_id)
        edge_id = edge.id
        source_context_token_ids = self._build_token_ids(
            token_ids,
            model_id=source_agent.model.id,
            device=source_agent.device,
        )
        # AgentRunner is a homogeneous-model control experiment. All logical
        # Agents share one physical model/tokenizer, so the target token ledger
        # is exactly the source token ledger; no cross-tokenization is performed.
        target_context_token_ids = self._build_token_ids(
            token_ids,
            model_id=target_agent.model.id,
            device=target_agent.device,
        )
        translated_past = self._build_algorithm_translated_past(
            edge=edge,
            source_past_key_values=source_past_key_values,
            source_context_token_ids=source_context_token_ids,
            target_context_token_ids=target_context_token_ids,
        )
        translated_tokens = get_past_seq_len(translated_past)
        if translated_tokens != source_tokens:
            raise ValueError(
                f"Pretranslated cache length mismatch on {edge_id}: "
                f"source_tokens={source_tokens} translated_tokens={translated_tokens}"
            )
        return edge_id, translated_past

    @torch.inference_mode()
    def _build_algorithm_translated_past(
        self,
        *,
        edge: Edge,
        source_past_key_values: PastKeyValues,
        source_context_token_ids: TokenIDs,
        target_context_token_ids: TokenIDs,
    ) -> PastKeyValues:
        tgt_spec = self.ctx.tp.get_model_spec(edge.tgt_id)
        target_model = self.ctx.tp.get_model(edge.tgt_id)

        if self.alg == "mot":
            from alg.mot.train import build_replayed_target_past

            translated_past, _ = build_replayed_target_past(
                self.ctx,
                source_past_key_values=source_past_key_values,
                source_context_token_ids=source_context_token_ids,
                target_context_token_ids=target_context_token_ids,
                source_model=self.ctx.tp.get_model(edge.src_id),
                target_model=target_model,
                src_node_id=edge.src_id,
                tgt_node_id=edge.tgt_id,
                tgt_spec=tgt_spec,
            )
            return translated_past

        if self.alg == "lsc":
            from alg.lsc.train import translate_layers

            return translate_layers(
                translator_pool=self.translator_pool,
                past_key_values=source_past_key_values,
                src_node_id=edge.src_id,
                tgt_node_id=edge.tgt_id,
                tgt_spec=tgt_spec,
            )

        if self.alg == "interlat":
            from alg.interlat.train import build_latent_conditioned_past, extract_interlat_source_hidden_states, translate_hidden_states

            source_tokens = get_past_seq_len(source_past_key_values)
            if int(source_context_token_ids.shape[1]) != source_tokens:
                raise ValueError(
                    f"InterLat prefix/token length mismatch on {edge.id}: "
                    f"prefix_tokens={int(source_context_token_ids.shape[1])} source_tokens={source_tokens}"
                )

            source_model = self.ctx.tp.get_model(edge.src_id)
            source_token_ids = source_context_token_ids.to(source_model.device)
            source_hidden_states = extract_interlat_source_hidden_states(
                source_model,
                source_token_ids,
            )
            translated_latents = translate_hidden_states(
                translator_pool=self.translator_pool,
                src_node_id=edge.src_id,
                tgt_node_id=edge.tgt_id,
                source_hidden_states=source_hidden_states,
            )
            translated_past = build_latent_conditioned_past(
                target_model,
                latent_prefix=translated_latents,
            )
            translated_tokens = get_past_seq_len(translated_past)
            if translated_tokens != source_tokens:
                raise ValueError(
                    f"InterLat translated cache length mismatch on {edge.id}: "
                    f"source_tokens={source_tokens} translated_tokens={translated_tokens}. "
                    "InterLat should communicate one latent per source prefix token; check the "
                    "InterLat translator output length and target model context window."
                )
            return translated_past

        if self.alg == "c2c-pr":
            from alg.c2c.train import translate_top_layers

            source_tokens = get_past_seq_len(source_past_key_values)
            if int(source_context_token_ids.shape[1]) != source_tokens:
                raise ValueError(
                    f"C2C-PR prefix/token length mismatch on {edge.id}: "
                    f"prefix_tokens={int(source_context_token_ids.shape[1])} source_tokens={source_tokens}"
                )
            native_target_past = extract_past_key_values(target_model, target_context_token_ids)
            translated_top_past = translate_top_layers(
                translator_pool=self.translator_pool,
                train_config=self.ctx.config,
                sharer_past_key_values=source_past_key_values,
                receiver_past_key_values=native_target_past,
                src_node_id=edge.src_id,
                tgt_node_id=edge.tgt_id,
                tgt_spec=tgt_spec,
            )
            translated_past = replace_top_layers(
                base_past_key_values=native_target_past,
                translated_top_past_key_values=translated_top_past,
            )
            translated_tokens = get_past_seq_len(translated_past)
            if translated_tokens != source_tokens:
                raise ValueError(
                    f"C2C-PR translated cache length mismatch on {edge.id}: "
                    f"source_tokens={source_tokens} translated_tokens={translated_tokens}"
                )
            return translated_past

        if self.alg == "kvcomm":
            from alg.kvcomm.train import build_replayed_target_past

            source_tokens = get_past_seq_len(source_past_key_values)
            if int(source_context_token_ids.shape[1]) != source_tokens:
                raise ValueError(
                    f"KVComm prefix/token length mismatch on {edge.id}: "
                    f"prefix_tokens={int(source_context_token_ids.shape[1])} source_tokens={source_tokens}"
                )
            translated_past = build_replayed_target_past(
                self.ctx,
                self.translator_pool,
                source_past_key_values=source_past_key_values,
                edge_id=edge.id,
                src_node_id=edge.src_id,
                tgt_node_id=edge.tgt_id,
            )
            translated_tokens = get_past_seq_len(translated_past)
            if translated_tokens != source_tokens:
                raise ValueError(
                    f"KVComm translated cache length mismatch on {edge.id}: "
                    f"source_tokens={source_tokens} translated_tokens={translated_tokens}"
                )
            return translated_past

        raise ValueError(f"Unsupported alg={self.alg!r}; expected one of {SUPPORTED_ALGS}")


    @torch.inference_mode()
    def refresh_pretranslated_cache(self, *, source_agent: Agent, target_agent: Agent) -> Dict[str, Any]:
        """Prepare the full translated target-side cache for source_agent -> target_agent.

        This is the only place where the selected algorithm translates a resident
        Agent cache. offload_cache() must not call the translator; it only slices
        this prepared cache according to the token-id delta computed by the runner.
        """
        if source_agent.past_key_values is None:
            raise ValueError(f"Source agent {source_agent.node_id} has no KV cache to pretranslate.")
        source_token_ids = list(source_agent.cache_token_ids)
        source_tokens = get_past_seq_len(source_agent.past_key_values)
        if len(source_token_ids) != source_tokens:
            raise ValueError(
                f"Source agent {source_agent.node_id} token-id ledger is not aligned with KV cache: "
                f"token_ids={len(source_token_ids)} past_tokens={source_tokens}"
            )

        edge = self._get_edge(source_agent.node_id, target_agent.node_id)
        edge_id = edge.id
        cached_ids = source_agent.pretranslated_token_ids_by_edge.get(edge_id)
        cached_past = source_agent.pretranslated_past_by_edge.get(edge_id)
        if cached_past is not None and cached_ids == source_token_ids:
            return {
                "edge_id": edge_id,
                "prepared_tokens": source_tokens,
                "cached": True,
            }

        edge_id, translated_past = self.build_pretranslated_past_for_edge(
            source_agent=source_agent,
            target_agent=target_agent,
            source_past_key_values=source_agent.past_key_values,
            source_token_ids=source_token_ids,
        )
        source_agent.set_pretranslated_cache(
            edge_id=edge_id,
            past_key_values=translated_past,
            cache_token_ids=source_token_ids,
        )
        return {
            "edge_id": edge_id,
            "prepared_tokens": source_tokens,
            "cached": False,
        }

    def _slice_pretranslated_cache_piece(
        self,
        *,
        source_agent: Agent,
        target_agent: Agent,
        prefix_tokens: int,
        expected_delta_tokens: int,
    ) -> Tuple[PastKeyValues, Dict[str, Any]]:
        edge = self._get_edge(source_agent.node_id, target_agent.node_id)
        edge_id = edge.id
        translated_full_past = source_agent.pretranslated_past_by_edge.get(edge_id)
        translated_token_ids = source_agent.pretranslated_token_ids_by_edge.get(edge_id)
        source_token_ids = list(source_agent.cache_token_ids)

        if translated_full_past is None or translated_token_ids is None:
            raise RuntimeError(
                f"No pretranslated cache is available for {edge_id}. "
                "Refresh the source Agent's outbound translation immediately after its cache changes "
                "and before calling offload_cache()."
            )
        if translated_token_ids != source_token_ids:
            raise RuntimeError(
                f"Stale pretranslated cache for {edge_id}: "
                f"prepared_tokens={len(translated_token_ids)} current_tokens={len(source_token_ids)}"
            )

        translated_piece = slice_past_suffix(translated_full_past, prefix_tokens)
        target_piece_tokens = get_past_seq_len(translated_piece)
        if target_piece_tokens != expected_delta_tokens:
            raise ValueError(
                f"Pretranslated delta slice length mismatch on {edge_id}: "
                f"expected_delta_tokens={expected_delta_tokens} target_piece_tokens={target_piece_tokens}"
            )

        return translated_piece, {
            "edge_id": edge_id,
            "piece_source_tokens": int(expected_delta_tokens),
            "piece_target_tokens": target_piece_tokens,
        }

    @torch.inference_mode()
    def offload_cache(
        self,
        *,
        source_agent: Agent,
        target_agent: Agent,
        target_tokens_before_replay: int,
        expected_delta_tokens: int,
        delta_prefix_matched: bool,
    ) -> Tuple[PastKeyValues, Dict[str, Any]]:
        """Replay a pretranslated KV delta into target_agent.

        No translation is performed here. The translated full cache must already
        exist on source_agent.pretranslated_past_by_edge; offload_cache() only
        slices the missing suffix and concatenates it to the target cache.
        """
        translated_piece, piece_meta = self._slice_pretranslated_cache_piece(
            source_agent=source_agent,
            target_agent=target_agent,
            prefix_tokens=target_tokens_before_replay,
            expected_delta_tokens=expected_delta_tokens,
        )

        if target_agent.past_key_values is None:
            replayed_target_past = translated_piece
        else:
            replayed_target_past = _concat_past_key_values(target_agent.past_key_values, translated_piece)

        source_piece_tokens = int(piece_meta.get("piece_source_tokens", 0))
        target_piece_tokens = int(piece_meta.get("piece_target_tokens", 0))
        return replayed_target_past, {
            "mode": "offload",
            "offload_kind": OFFLOAD_KIND_DELTA,
            "edge_id": piece_meta.get("edge_id"),
            "tokens_sent": source_piece_tokens,
            "tokens_received": source_piece_tokens,
            "target_piece_tokens": target_piece_tokens,
            "target_tokens_before_replay": int(target_tokens_before_replay),
            "target_tokens_after_replay": get_past_seq_len(replayed_target_past),
            "expected_delta_tokens": int(expected_delta_tokens),
            "delta_prefix_matched": bool(delta_prefix_matched),
        }



class AgentRunner:
    def __init__(
        self,
        *,
        ctx: Context,
        translator_pool,
        alg: str,
        max_turns: int = 4,
        generation_max_new_tokens: int = 48,
        max_prompt_tokens: Optional[int] = None,
        seed: int = 42,
        log_turns: bool = True,
        log_max_chars: int = 600,
        cache_mode: str = CACHE_MODE_RETAIN,
        agent_count: Optional[int] = None,
    ) -> None:
        total_nodes = len(ctx.nodes)
        if not _is_homogeneous_model_pool(ctx.nodes):
            raise ValueError(
                "AgentRunner is a homogeneous-model control experiment and does not support "
                "heterogeneous model pools. Use the same model_id for every checkpoint node."
            )
        physical_models = {id(ctx.tp.get_model(node.id)) for node in ctx.nodes}
        if len(physical_models) != 1:
            raise RuntimeError(
                "AgentRunner requires all logical checkpoint nodes to share one physical Model instance."
            )
        resolved_alg = normalize_agent_runner_alg(alg)
        if cache_mode not in SUPPORTED_CACHE_MODES:
            raise ValueError(f"Unsupported cache_mode={cache_mode!r}; expected one of {SUPPORTED_CACHE_MODES}")
        if resolved_alg in RETAIN_ONLY_ALGS and cache_mode != CACHE_MODE_RETAIN:
            raise ValueError(f"alg={resolved_alg!r} supports only cache_mode='retain'.")
        resolved_agent_count = resolve_agent_count(agent_count, total_nodes)

        self.ctx = ctx
        self.translator_pool = translator_pool
        self.alg = resolved_alg
        self.max_turns = int(max_turns)
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be at least 1, got {self.max_turns}")
        self.generation_max_new_tokens = int(generation_max_new_tokens)
        self.max_prompt_tokens = max_prompt_tokens
        self.seed = int(seed)
        self.log_turns = bool(log_turns)
        self.log_max_chars = max(80, int(log_max_chars))
        self.cache_mode = cache_mode
        self.agent_count = resolved_agent_count
        self.device = ctx.config.device
        self.cache_translator = KVCacheTranslationAdapter(ctx=ctx, translator_pool=translator_pool, alg=alg)
        self._canonical_model_node_id = ctx.nodes[0].id

        if self.agent_count <= total_nodes:
            self.active_nodes = list(ctx.nodes[: self.agent_count])
        else:
            model_id = ctx.nodes[0].model_id
            self.active_nodes = [
                Node(id=index_to_node_id(index), model_id=model_id)
                for index in range(self.agent_count)
            ]
        self.node_ids = [node.id for node in self.active_nodes]
        stop_sequences = tuple(f"\nAgent {node.id}:" for node in self.active_nodes) + (
            "\n### Instruction:",
            "\n### Passage:",
            "\n### Question:",
            "\nQuestion:",
            "\nPassage:",
        )
        self.agent_sequence: List[Agent] = []
        self.logical_to_physical_node_id: Dict[str, str] = {}
        for index, node in enumerate(self.active_nodes):
            agent_cls = HubAgent if index == 0 else Agent
            physical_node_id = self._canonical_model_node_id
            self.logical_to_physical_node_id[node.id] = physical_node_id
            self.agent_sequence.append(
                agent_cls(
                    node_id=node.id,
                    model=ctx.tp.get_model(physical_node_id),
                    device=self.device,
                    max_new_tokens=self.generation_max_new_tokens,
                    stop_sequences=stop_sequences,
                    max_prompt_tokens=max_prompt_tokens,
                )
            )

        self.hub_agent: HubAgent = self.agent_sequence[0]  # type: ignore[assignment]
        self.non_hub_agents = self.agent_sequence[1:]
        self.agents = {agent.node_id: agent for agent in self.agent_sequence}

        # Backward-compatible aliases for older two-agent experiments/tests.
        # agent_count=1 is a valid Hub-only baseline, so agent_b is absent there.
        self.agent_a = self.hub_agent
        self.agent_b = self.agent_sequence[1] if len(self.agent_sequence) > 1 else None

        # For non-hub -> non-hub logical handoffs, the physical path is
        # source -> hub -> target. The second hop's hub -> target cache is
        # precomputed right after the source generation, before any offload starts,
        # then installed on the hub after the first hop updates the hub cache.
        self._pending_pretranslated_second_hops: Dict[Tuple[str, str], Tuple[str, PastKeyValues, List[int]]] = {}
        self._peak_memory_breakdown_bytes: Optional[GPUMemoryBreakdownBytes] = None

    @classmethod
    def from_checkpoint(cls, config: AgentRunnerConfig) -> "AgentRunner":
        resolved_alg = normalize_agent_runner_alg(config.alg)
        if resolved_alg in RETAIN_ONLY_ALGS and config.cache_mode != CACHE_MODE_RETAIN:
            raise ValueError(f"alg={resolved_alg!r} supports only cache_mode='retain'.")
        if not config.checkpoint_dir_path:
            raise ValueError("checkpoint_dir_path is required")
        train_config_path = get_train_config_path(config.checkpoint_dir_path)
        if not train_config_path.exists():
            raise FileNotFoundError(f"Train config not found: {train_config_path}")
        train_payload = read_json(train_config_path)
        all_nodes, all_edges = build_nodes_and_edges(train_payload["model_ids"], train_payload["model_directions"])
        if not _is_homogeneous_model_pool(all_nodes):
            raise ValueError(
                "AgentRunner is a homogeneous-model control experiment and does not support "
                "heterogeneous checkpoints. All model_ids in the training config must be identical."
            )
        resolve_agent_count(config.agent_count, len(all_nodes))

        train_mod = importlib.import_module(TRAIN_MODULE_BY_ALG[resolved_alg])
        loaded = train_mod.load_translator_pool_from_checkpoint(
            checkpoint_dir_path=config.checkpoint_dir_path,
            device_override=resolve_device(config.device),
        )
        ctx, translator_pool, *_ = loaded
        set_seed(config.seed)
        return cls(
            ctx=ctx,
            translator_pool=translator_pool,
            alg=resolved_alg,
            max_turns=config.max_turns,
            generation_max_new_tokens=config.generation_max_new_tokens,
            max_prompt_tokens=config.max_prompt_tokens,
            seed=config.seed,
            log_turns=config.log_turns,
            log_max_chars=config.log_max_chars,
            cache_mode=config.cache_mode,
            agent_count=config.agent_count,
        )

    @staticmethod
    def _final_answer_instruction() -> str:
        return "Return only one of: FINAL: yes or FINAL: no"

    @staticmethod
    def _initial_agent_instruction(agent_id: str) -> str:
        return (
            f"You are Agent {agent_id}, the first participant in a collaborative discussion with shared memory. "
            "No solution has been proposed yet. Answer the StrategyQA question using your own knowledge and reasoning. "
            "Give a concise justification, then end your response with 'Answer: yes' or 'Answer: no'."
        )

    @staticmethod
    def _ordinary_agent_instruction(agent_id: str, previous_agent_id: str) -> str:
        return (
            f"You are Agent {agent_id} in a collaborative discussion with shared memory. You can use the full "
            "discussion history from all previous Agents. Improve the current solution, which is the most recent "
            f"solution proposed by Agent {previous_agent_id}. If you agree with the current solution, answer with "
            "[AGREE]. Otherwise answer with [DISAGREE], explain why, and provide an improved solution. "
            "Let's think step-by-step. End with 'Answer: yes' or 'Answer: no'."
        )

    @staticmethod
    def _judge_instruction(previous_agent_id: str) -> str:
        del previous_agent_id
        return (
            "You are the Judge. Provide a decision on the listed solutions and combine them into a single answer "
            "to solve the StrategyQA question. Only answer with the final solution. "
            f"{AgentRunner._final_answer_instruction()}"
        )

    @staticmethod
    def _format_judge_solutions(solutions: Sequence[Tuple[str, str]]) -> str:
        return "\n".join(
            f"Solution {index} (Agent {agent_id}): {response.strip()}"
            for index, (agent_id, response) in enumerate(solutions, start=1)
        )

    @staticmethod
    def _latest_agent_solutions(turns: Sequence[AgentTurnRecord]) -> List[Tuple[str, str]]:
        latest = {}
        order = []
        for turn in turns:
            if turn.agent_id not in latest:
                order.append(turn.agent_id)
            latest[turn.agent_id] = turn.response
        return [(agent_id, latest[agent_id]) for agent_id in order]

    @staticmethod
    def _build_plain_initial_prompt(
        context: str,
        question: str,
        *,
        agent_id: str = "A",
        is_final_turn: bool = False,
    ) -> str:
        del is_final_turn
        instruction = AgentRunner._initial_agent_instruction(agent_id)
        context_block = f"### Context:\n{context.strip()}\n" if context.strip() else ""
        return (
            f"### Instruction: {instruction}\n"
            f"{context_block}"
            "### Question:\n"
            f"{question.strip()}\n"
            "### Response:\n"
        )

    @staticmethod
    def _build_plain_followup_prompt(
        question: str,
        *,
        agent_id: str,
        previous_agent_id: str,
        is_final_turn: bool = False,
        judge_solutions: Sequence[Tuple[str, str]] = (),
    ) -> str:
        instruction = (
            AgentRunner._judge_instruction(previous_agent_id)
            if is_final_turn
            else AgentRunner._ordinary_agent_instruction(agent_id, previous_agent_id)
        )
        solutions_block = (
            f"### Solutions:\n{AgentRunner._format_judge_solutions(judge_solutions)}\n"
            if is_final_turn and judge_solutions
            else ""
        )
        return (
            "\n### Instruction:\n"
            f"{instruction}\n"
            "### Question:\n"
            f"{question.strip()}\n"
            f"{solutions_block}"
            "### Response:\n"
        )

    @staticmethod
    def _explicit_chat_template(agent: Agent):
        """Return a tokenizer-provided chat template, never a library default."""
        template = getattr(agent.model.tokenizer, "chat_template", None)
        if isinstance(template, dict):
            template = template.get("default") or next(iter(template.values()), None)
        if isinstance(template, str) and template.strip():
            return template
        return None

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are participating in a collaborative discussion to solve a StrategyQA yes/no question. "
            "The discussion uses shared memory: each participant can use the full discussion history produced so far."
        )

    @staticmethod
    def _build_initial_user_content(
        context: str,
        question: str,
        *,
        agent_id: str = "A",
        is_final_turn: bool = False,
    ) -> str:
        del is_final_turn
        instruction = AgentRunner._initial_agent_instruction(agent_id)
        context_block = f"### Context:\n{context.strip()}\n" if context.strip() else ""
        return (
            "### Instruction:\n"
            f"{instruction}\n"
            f"{context_block}"
            "### Question:\n"
            f"{question.strip()}"
        )

    @staticmethod
    def _build_followup_user_content(
        question: str,
        *,
        agent_id: str,
        previous_agent_id: str,
        is_final_turn: bool = False,
        judge_solutions: Sequence[Tuple[str, str]] = (),
    ) -> str:
        instruction = (
            AgentRunner._judge_instruction(previous_agent_id)
            if is_final_turn
            else AgentRunner._ordinary_agent_instruction(agent_id, previous_agent_id)
        )
        solutions_block = (
            f"\n### Solutions:\n{AgentRunner._format_judge_solutions(judge_solutions)}"
            if is_final_turn and judge_solutions
            else ""
        )
        return (
            "### Instruction:\n"
            f"{instruction}\n"
            "### Question:\n"
            f"{question.strip()}"
            f"{solutions_block}"
        )

    @staticmethod
    def _strip_leading_bos_for_continuation(tokenizer, rendered: str) -> str:
        bos_token = getattr(tokenizer, "bos_token", None)
        if isinstance(bos_token, str) and bos_token and rendered.startswith(bos_token):
            return rendered[len(bos_token) :]
        return rendered

    @staticmethod
    def _render_chat_prompt(
        agent: Agent,
        user_content: str,
        *,
        continuation: bool,
    ) -> Optional[str]:
        tokenizer = agent.model.tokenizer
        chat_template = AgentRunner._explicit_chat_template(agent)
        if chat_template is None:
            return None

        if continuation:
            # The existing KV cache already contains the prior conversation and
            # its generated EOS/end-of-turn token. Append only a new user turn.
            message_sets = [[{"role": "user", "content": user_content.strip()}]]
        else:
            # Some checkpoints (for example Gemma-family templates) do not accept
            # a system role. Try the canonical system+user form first, then fold
            # the system instruction into the user message without model-specific
            # branching.
            message_sets = [
                [
                    {"role": "system", "content": AgentRunner._system_prompt()},
                    {"role": "user", "content": user_content.strip()},
                ],
                [
                    {
                        "role": "user",
                        "content": f"{AgentRunner._system_prompt()}\n\n{user_content.strip()}",
                    }
                ],
            ]

        for messages in message_sets:
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    chat_template=chat_template,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if rendered:
                    rendered = str(rendered)
                    if continuation:
                        rendered = AgentRunner._strip_leading_bos_for_continuation(tokenizer, rendered)
                    return rendered
            except Exception as error:
                logging.debug("Chat-template rendering failed; trying fallback/plain prompt: %s", error)
        return None

    def _format_agent_prompt(
        self,
        agent: Agent,
        user_content: str,
        fallback_prompt: str,
        *,
        continuation: bool,
    ) -> str:
        rendered = self._render_chat_prompt(agent, user_content, continuation=continuation)
        return rendered if rendered else fallback_prompt

    @staticmethod
    def build_initial_prompt(
        context: str,
        question: str,
        *,
        hub_agent_id: str = "A",
        agent_count: int = 2,
        is_final_turn: bool = False,
    ) -> str:
        del agent_count
        return AgentRunner._build_plain_initial_prompt(
            context,
            question,
            agent_id=hub_agent_id,
            is_final_turn=is_final_turn,
        )

    def build_initial_prompt_for_agent(
        self,
        agent: Agent,
        context: str,
        question: str,
        *,
        hub_agent_id: str = "A",
        agent_count: int = 2,
        is_final_turn: bool = False,
    ) -> str:
        del hub_agent_id, agent_count
        user_content = self._build_initial_user_content(
            context,
            question,
            agent_id=agent.node_id,
            is_final_turn=is_final_turn,
        )
        return self._format_agent_prompt(
            agent,
            user_content,
            self._build_plain_initial_prompt(
                context,
                question,
                agent_id=agent.node_id,
                is_final_turn=is_final_turn,
            ),
            continuation=False,
        )

    def build_followup_prompt(
        self,
        agent_id: str,
        turn_index: int,
        question: str,
        *,
        is_final_turn: bool = False,
        judge_solutions: Sequence[Tuple[str, str]] = (),
    ) -> str:
        previous_agent_id = self._agent_for_turn(max(0, turn_index - 1)).node_id
        user_content = self._build_followup_user_content(
            question,
            agent_id=agent_id,
            previous_agent_id=previous_agent_id,
            is_final_turn=is_final_turn,
            judge_solutions=judge_solutions,
        )
        agent = self.agents.get(agent_id)
        if agent is not None:
            return self._format_agent_prompt(
                agent,
                user_content,
                self._build_plain_followup_prompt(
                    question,
                    agent_id=agent_id,
                    previous_agent_id=previous_agent_id,
                    is_final_turn=is_final_turn,
                    judge_solutions=judge_solutions,
                ),
                continuation=True,
            )
        return self._build_plain_followup_prompt(
            question,
            agent_id=agent_id,
            previous_agent_id=previous_agent_id,
            is_final_turn=is_final_turn,
            judge_solutions=judge_solutions,
        )

    @staticmethod
    def _append_turn_to_transcript(transcript: str, agent_id: str, response: str) -> str:
        clean_response = response.strip() or "[empty]"
        return f"{transcript}{clean_response}\n"

    @staticmethod
    def extract_final_answer(transcript: str, fallback_response: str) -> str:
        del transcript
        final_matches = list(
            re.finditer(r"FINAL\s*:\s*(yes|no)\b", fallback_response or "", flags=re.IGNORECASE)
        )
        if final_matches:
            return final_matches[-1].group(1).lower()
        binary_matches = list(re.finditer(r"\b(yes|no)\b", fallback_response or "", flags=re.IGNORECASE))
        if binary_matches:
            return binary_matches[-1].group(1).lower()
        return postprocess_generated_answer((fallback_response or "").strip())

    @staticmethod
    def _preview_text(text: str, max_chars: int) -> str:
        clean = re.sub(r"\s+", " ", (text or "").strip())
        if len(clean) <= max_chars:
            return clean
        return clean[: max(0, max_chars - 3)] + "..."

    def _count_resident_cache_agents(self) -> int:
        return sum(1 for agent in self.agent_sequence if agent.past_key_values is not None)

    def _agent_for_turn(self, turn_index: int) -> Agent:
        return self.agent_sequence[turn_index % len(self.agent_sequence)]

    def _update_peak_memory_breakdown(self) -> None:
        measured = measure_gpu_memory_breakdown_bytes(
            self.device,
            models=self.ctx.tp.models.values(),
            translator_pool=self.translator_pool,
        )
        if measured is None:
            return
        previous = self._peak_memory_breakdown_bytes
        if previous is None:
            self._peak_memory_breakdown_bytes = measured
            return
        self._peak_memory_breakdown_bytes = GPUMemoryBreakdownBytes(
            model_bytes=max(previous.model_bytes, measured.model_bytes),
            translator_bytes=max(previous.translator_bytes, measured.translator_bytes),
            kv_bytes=max(previous.kv_bytes, measured.kv_bytes),
        )

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
        turn_order = "->".join(self.node_ids + ([self.node_ids[0]] if len(self.node_ids) > 1 else []))
        logging.info(
            "[AgentRunner][example=%s] start | mode=%s | alg=%s | agents=%s | judge=%s | discussion=memory | physical_topology=star | turn_order=%s | "
            "question=%s | gold=%s",
            label,
            self.cache_mode,
            self.alg,
            len(self.agent_sequence),
            self.hub_agent.node_id,
            turn_order,
            self._preview_text(question, self.log_max_chars),
            list(gold_answers)[:3],
        )

    def _log_turn(self, *, example_index: Optional[int], turn_index: int, record: AgentTurnRecord) -> None:
        if not self.log_turns:
            return
        label = "?" if example_index is None else str(example_index)
        logging.info(
            "[AgentRunner][example=%s][turn=%d] mode=%s | agent=%s | hub=%s | "
            "translated_edge_id=%s(%s) | offload_edge_id=%s(%s) | "
            "tokens_before=%d | tokens_after=%d | tokens_prompt=%d | tokens_completion=%d | "
            "tokens_received=%d | tokens_sent=%d | tokens_sent_check_passed=%s",
            label,
            turn_index,
            record.cache_mode,
            record.agent_id,
            record.is_hub,
            record.translated_edge_id or "null",
            record.translated_offload_kind or "none",
            record.offload_edge_id or "null",
            record.offload_kind or "none",
            record.tokens_before,
            record.tokens_after,
            record.tokens_prompt,
            record.tokens_completion,
            record.tokens_received,
            record.tokens_sent,
            record.tokens_sent_check_passed,
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
            self._preview_text(record.response, self.log_max_chars),
        )

    def _log_example_end(self, *, example_index: Optional[int], prediction: str, accuracy: float) -> None:
        if not self.log_turns:
            return
        label = "?" if example_index is None else str(example_index)
        logging.info(
            "[AgentRunner][example=%s] end | mode=%s | prediction=%s | accuracy=%.4f",
            label,
            self.cache_mode,
            self._preview_text(prediction, self.log_max_chars),
            accuracy,
        )

    def _should_clear_source_after_offload(self, source_agent: Agent) -> bool:
        # In free mode, the non-hub agent that just transmitted its newly generated
        # cache delta is freed. The hub stays resident so it can always receive and
        # accumulate deltas from the other agent(s).
        return self.cache_mode == CACHE_MODE_FREE and source_agent.node_id != self.hub_agent.node_id

    def _prepare_outgoing_route_translation(self, *, source_agent: Agent, logical_target_agent: Agent) -> None:
        """Pretranslate the physical star-topology route before the next handoff.

        Direct hub-involved handoffs prepare source->target. Non-hub->non-hub
        handoffs prepare both physical hops in advance: source->hub is stored on
        the source Agent, and the future hub->target cache is kept pending until
        the first hop installs the corresponding hub cache. No translator call is
        made inside offload_cache().
        """
        self._pending_pretranslated_second_hops.pop((source_agent.node_id, logical_target_agent.node_id), None)
        if source_agent.past_key_values is None:
            return

        source_is_hub = source_agent.node_id == self.hub_agent.node_id
        target_is_hub = logical_target_agent.node_id == self.hub_agent.node_id
        if source_is_hub or target_is_hub:
            self.cache_translator.refresh_pretranslated_cache(
                source_agent=source_agent,
                target_agent=logical_target_agent,
            )
            return

        # First physical hop: source non-hub -> hub.
        first_meta = self.cache_translator.refresh_pretranslated_cache(
            source_agent=source_agent,
            target_agent=self.hub_agent,
        )
        first_edge_id = str(first_meta["edge_id"])
        first_full_hub_past = source_agent.pretranslated_past_by_edge[first_edge_id]

        # Build the exact future hub cache that will exist after the first hop by
        # slicing the already-pretranslated source->hub cache. This is still route
        # preparation, not offload; the actual handoff later only slices/copies.
        prefix_tokens, expected_delta_tokens, _ = self._build_missing_cache_delta(
            source_agent=source_agent,
            target_agent=self.hub_agent,
        )
        first_delta_piece = slice_past_suffix(first_full_hub_past, prefix_tokens)
        if get_past_seq_len(first_delta_piece) != expected_delta_tokens:
            raise ValueError(
                f"Prepared source->hub delta length mismatch on {first_edge_id}: "
                f"expected_delta_tokens={expected_delta_tokens} "
                f"piece_tokens={get_past_seq_len(first_delta_piece)}"
            )
        if self.hub_agent.past_key_values is None:
            future_hub_past = first_delta_piece
        else:
            future_hub_past = _concat_past_key_values(self.hub_agent.past_key_values, first_delta_piece)

        # Second physical hop: future hub -> logical target.
        second_edge_id, second_full_target_past = self.cache_translator.build_pretranslated_past_for_edge(
            source_agent=self.hub_agent,
            target_agent=logical_target_agent,
            source_past_key_values=future_hub_past,
            source_token_ids=source_agent.cache_token_ids,
        )
        self._pending_pretranslated_second_hops[(source_agent.node_id, logical_target_agent.node_id)] = (
            second_edge_id,
            second_full_target_past,
            list(source_agent.cache_token_ids),
        )

    def _build_missing_cache_delta(
        self,
        *,
        source_agent: Agent,
        target_agent: Agent,
    ) -> Tuple[int, int, bool]:
        """Return token-only metadata for the source suffix absent from target_agent.

        past_key_values does not contain token ids. Each Agent carries a
        cache_token_ids ledger aligned with its resident KV cache. Delta size is
        computed only from these ledgers; no KV delta is sliced or translated here.
        """
        if source_agent.past_key_values is None:
            raise ValueError(f"Source agent {source_agent.node_id} has no KV cache to offload.")
        if len(source_agent.cache_token_ids) != get_past_seq_len(source_agent.past_key_values):
            raise ValueError(
                f"Source agent {source_agent.node_id} token-id ledger is not aligned with KV cache: "
                f"token_ids={len(source_agent.cache_token_ids)} past_tokens={get_past_seq_len(source_agent.past_key_values)}"
            )
        if target_agent.past_key_values is not None and len(target_agent.cache_token_ids) != get_past_seq_len(target_agent.past_key_values):
            raise ValueError(
                f"Target agent {target_agent.node_id} token-id ledger is not aligned with KV cache: "
                f"token_ids={len(target_agent.cache_token_ids)} past_tokens={get_past_seq_len(target_agent.past_key_values)}"
            )

        source_ids = list(source_agent.cache_token_ids)
        target_ids = list(target_agent.cache_token_ids) if target_agent.past_key_values is not None else []
        if not target_ids:
            prefix_tokens = 0
            prefix_matched = True
        elif source_ids[: len(target_ids)] == target_ids:
            prefix_tokens = len(target_ids)
            prefix_matched = True
        else:
            raise ValueError(
                f"Cannot offload a token-id delta because target cache is not a prefix of source cache "
                f"on {source_agent.node_id}->{target_agent.node_id}: "
                f"source_tokens={len(source_ids)} target_tokens={len(target_ids)}"
            )

        expected_delta_tokens = len(source_ids) - prefix_tokens
        if expected_delta_tokens <= 0:
            raise ValueError(
                f"No KV delta to offload on {source_agent.node_id}->{target_agent.node_id}: "
                f"source_tokens={len(source_ids)} target_prefix_tokens={prefix_tokens}"
            )
        return prefix_tokens, expected_delta_tokens, prefix_matched

    def _offload_into(
        self,
        *,
        source_agent: Agent,
        target_agent: Agent,
        target_tokens_before_replay: int,
        expected_delta_tokens: int,
        delta_prefix_matched: bool,
    ) -> Tuple[Dict[str, Any], bool]:
        translated_past, metadata = self.cache_translator.offload_cache(
            source_agent=source_agent,
            target_agent=target_agent,
            target_tokens_before_replay=target_tokens_before_replay,
            expected_delta_tokens=expected_delta_tokens,
            delta_prefix_matched=delta_prefix_matched,
        )
        # After replay, the target owns the same logical token-id span that the
        # source had at handoff time. Future deltas are computed from token ids.
        target_agent.set_replayed_cache(
            translated_past,
            cache_token_ids=source_agent.cache_token_ids,
        )

        cleared = False
        if self._should_clear_source_after_offload(source_agent):
            if source_agent.past_key_values is not None:
                source_agent.clear_kv_cache(empty_cuda_cache=True)
                cleared = True
        return metadata, cleared

    def _offload_delta_hop(
        self,
        *,
        source_agent: Agent,
        target_agent: Agent,
    ) -> Tuple[Dict[str, Any], bool]:
        """Run one physical star-topology offload hop.

        Delta still means the KV suffix that exists in source_agent but does not
        exist in target_agent. If target_agent has no resident cache, this delta
        naturally spans source_agent's whole resident cache without introducing
        a separate full-offload mode.
        """
        target_tokens_before_replay, expected_delta_tokens, delta_prefix_matched = self._build_missing_cache_delta(
            source_agent=source_agent,
            target_agent=target_agent,
        )
        return self._offload_into(
            source_agent=source_agent,
            target_agent=target_agent,
            target_tokens_before_replay=target_tokens_before_replay,
            expected_delta_tokens=expected_delta_tokens,
            delta_prefix_matched=delta_prefix_matched,
        )

    def _star_offload_to_turn_target(
        self,
        *,
        source_agent: Agent,
        target_agent: Agent,
    ) -> Tuple[Agent, Dict[str, Any], bool, Agent, Dict[str, Any]]:
        """Route KV handoff through the hub when both endpoints are non-hub.

        Returns:
            record_target_agent: the physical target of source_agent's outgoing
                offload, used for source_agent's turn record.
            source_offload_meta: metadata for source_agent's physical outgoing hop.
            source_cache_cleared: whether source_agent's cache was cleared.
            incoming_from_agent: the physical source that supplied target_agent's
                replayed cache for the next generation.
            incoming_meta: metadata for the hop that actually entered target_agent.
        """
        source_is_hub = source_agent.node_id == self.hub_agent.node_id
        target_is_hub = target_agent.node_id == self.hub_agent.node_id

        if source_is_hub or target_is_hub:
            offload_meta, cleared = self._offload_delta_hop(
                source_agent=source_agent,
                target_agent=target_agent,
            )
            return target_agent, offload_meta, cleared, source_agent, offload_meta

        # Non-hub -> non-hub is never direct in the star topology. The physical
        # route is source -> Hub, then Hub -> target. The second hop cache must
        # already have been prepared after the source generation; this method only
        # installs that prepared cache after the hub receives the first hop.
        pending_key = (source_agent.node_id, target_agent.node_id)
        pending_second_hop = self._pending_pretranslated_second_hops.pop(pending_key, None)
        if pending_second_hop is None:
            raise RuntimeError(
                f"Missing pretranslated second-hop cache for {source_agent.node_id}->"
                f"{self.hub_agent.node_id}->{target_agent.node_id}."
            )

        first_meta, source_cleared = self._offload_delta_hop(
            source_agent=source_agent,
            target_agent=self.hub_agent,
        )

        second_edge_id, second_full_target_past, second_token_ids = pending_second_hop
        if list(self.hub_agent.cache_token_ids) != list(second_token_ids):
            raise RuntimeError(
                f"Prepared second-hop cache is stale for {second_edge_id}: "
                f"prepared_tokens={len(second_token_ids)} hub_tokens={len(self.hub_agent.cache_token_ids)}"
            )
        self.hub_agent.set_pretranslated_cache(
            edge_id=second_edge_id,
            past_key_values=second_full_target_past,
            cache_token_ids=second_token_ids,
        )
        second_meta, _ = self._offload_delta_hop(
            source_agent=self.hub_agent,
            target_agent=target_agent,
        )
        return self.hub_agent, first_meta, source_cleared, self.hub_agent, second_meta

    def _tokens_sent_check_passed(self, record: AgentTurnRecord, offload_meta: Dict[str, Any]) -> bool:
        # Every handoff is a delta. The expected amount is
        # source_cache_tokens_after_generation - target_cache_tokens_before_replay.
        tokens_sent = int(offload_meta.get("tokens_sent", 0))
        expected_delta_tokens = int(offload_meta.get("expected_delta_tokens", 0))
        return tokens_sent == expected_delta_tokens

    def _apply_offload_metadata_to_record(
        self,
        record: AgentTurnRecord,
        *,
        target_agent: Agent,
        offload_meta: Dict[str, Any],
        source_was_freed: bool,
    ) -> None:
        """Record the outgoing handoff on the agent that actually sent the KV cache."""
        edge_id = str(offload_meta.get("edge_id", "")) or None
        offload_kind = str(offload_meta.get("offload_kind", "")) or None
        tokens_sent = int(offload_meta.get("tokens_sent", 0))
        del target_agent, source_was_freed
        record.offload_edge_id = edge_id
        record.offload_kind = offload_kind
        record.tokens_sent = tokens_sent
        record.tokens_sent_check_passed = self._tokens_sent_check_passed(record, offload_meta)

    def _run_offload_turns(
        self,
        *,
        transcript: str,
        initial_generation: AgentGeneration,
        turns: List[AgentTurnRecord],
        question: str,
        example_index: Optional[int],
    ) -> Tuple[str, str]:
        last_response = initial_generation.text
        normal_turn_count = self.max_turns
        last_normal_turn_index = normal_turn_count - 1

        # max_turns counts ordinary collaborative generations. After those
        # generations complete, the Hub always performs one separate FINAL turn.
        # Each turn record has two distinct directions:
        #   - translated_*: the incoming KV delta used by this agent's generation.
        #   - offload_*:   the physical outgoing KV delta sent by this agent after generation.
        # Handoffs use a hub-centered star topology. If two consecutive logical
        # turns use the same Agent (for example agent_count=1, or the mandatory
        # final Hub turn immediately follows a normal Hub turn), no KV handoff is
        # needed because that Agent already owns the current resident cache.
        for turn_index in range(1, normal_turn_count):
            current_source = self._agent_for_turn(turn_index - 1)
            current_target = self._agent_for_turn(turn_index)
            source_record = turns[turn_index - 1]

            edge_id: Optional[str] = None
            incoming_offload_kind: Optional[str] = None
            tokens_received = 0
            if current_source.node_id != current_target.node_id:
                (
                    record_offload_target,
                    source_offload_meta,
                    source_cache_cleared,
                    incoming_from_agent,
                    incoming_meta,
                ) = self._star_offload_to_turn_target(
                    source_agent=current_source,
                    target_agent=current_target,
                )

                self._apply_offload_metadata_to_record(
                    source_record,
                    target_agent=record_offload_target,
                    offload_meta=source_offload_meta,
                    source_was_freed=source_cache_cleared,
                )
                edge_id = str(incoming_meta.get("edge_id", "")) or None
                incoming_offload_kind = str(incoming_meta.get("offload_kind", "")) or None
                tokens_received = int(incoming_meta.get("tokens_received", incoming_meta.get("tokens_sent", 0)))
                del incoming_from_agent

            self._update_peak_memory_breakdown()
            self._log_turn(example_index=example_index, turn_index=turn_index - 1, record=source_record)

            prompt = self.build_followup_prompt(
                current_target.node_id,
                turn_index,
                question,
                is_final_turn=False,
            )
            generation = current_target.generate_response(prompt)

            if turn_index < last_normal_turn_index:
                next_target = self._agent_for_turn(turn_index + 1)
                if current_target.node_id != next_target.node_id:
                    self._prepare_outgoing_route_translation(
                        source_agent=current_target,
                        logical_target_agent=next_target,
                    )
            elif current_target.node_id != self.hub_agent.node_id:
                # Prepare the mandatory final handoff while the source cache is
                # still resident. The handoff itself is performed below.
                self._prepare_outgoing_route_translation(
                    source_agent=current_target,
                    logical_target_agent=self.hub_agent,
                )

            self._update_peak_memory_breakdown()
            transcript = self._append_turn_to_transcript(transcript + prompt, current_target.node_id, generation.text)
            turns.append(
                self._turn_record(
                    generation,
                    translated_edge_id=edge_id,
                    translated_offload_kind=incoming_offload_kind,
                    tokens_received=tokens_received,
                )
            )
            last_response = generation.text

        # Always perform one separate final Hub generation after exactly max_turns
        # ordinary collaborative generations, even when the last ordinary turn
        # was already produced by the Hub.
        final_turn_index = len(turns)
        current_source = self._agent_for_turn(last_normal_turn_index)
        source_record = turns[-1]
        edge_id = None
        incoming_offload_kind = None
        tokens_received = 0

        if current_source.node_id != self.hub_agent.node_id:
            (
                record_offload_target,
                source_offload_meta,
                source_cache_cleared,
                incoming_from_agent,
                incoming_meta,
            ) = self._star_offload_to_turn_target(
                source_agent=current_source,
                target_agent=self.hub_agent,
            )
            self._apply_offload_metadata_to_record(
                source_record,
                target_agent=record_offload_target,
                offload_meta=source_offload_meta,
                source_was_freed=source_cache_cleared,
            )
            edge_id = str(incoming_meta.get("edge_id", "")) or None
            incoming_offload_kind = str(incoming_meta.get("offload_kind", "")) or None
            tokens_received = int(incoming_meta.get("tokens_received", incoming_meta.get("tokens_sent", 0)))
            del incoming_from_agent

        self._update_peak_memory_breakdown()
        self._log_turn(example_index=example_index, turn_index=final_turn_index - 1, record=source_record)

        final_prompt = self.build_followup_prompt(
            self.hub_agent.node_id,
            final_turn_index,
            question,
            is_final_turn=True,
            judge_solutions=self._latest_agent_solutions(turns),
        )
        final_generation = self.hub_agent.generate_response(final_prompt)
        self._update_peak_memory_breakdown()
        transcript = self._append_turn_to_transcript(
            transcript + final_prompt,
            self.hub_agent.node_id,
            final_generation.text,
        )
        turns.append(
            self._turn_record(
                final_generation,
                translated_edge_id=edge_id,
                translated_offload_kind=incoming_offload_kind,
                tokens_received=tokens_received,
            )
        )
        last_response = final_generation.text

        # The final Hub turn has no following offload inside this run.
        self._log_turn(example_index=example_index, turn_index=final_turn_index, record=turns[-1])
        return transcript, last_response

    def _run_retain_turns(
        self,
        *,
        transcript: str,
        initial_generation: AgentGeneration,
        turns: List[AgentTurnRecord],
        question: str,
        example_index: Optional[int],
    ) -> Tuple[str, str]:
        return self._run_offload_turns(
            transcript=transcript,
            initial_generation=initial_generation,
            turns=turns,
            question=question,
            example_index=example_index,
        )

    def _run_free_turns(
        self,
        *,
        transcript: str,
        initial_generation: AgentGeneration,
        turns: List[AgentTurnRecord],
        question: str,
        example_index: Optional[int],
    ) -> Tuple[str, str]:
        return self._run_offload_turns(
            transcript=transcript,
            initial_generation=initial_generation,
            turns=turns,
            question=question,
            example_index=example_index,
        )

    def _run_example_impl(
        self,
        *,
        context: str,
        question: str,
        gold_answers: Sequence[str],
        example_index: Optional[int] = None,
    ) -> AgentRunnerResult:
        self._peak_memory_breakdown_bytes = None
        for agent in self.agent_sequence:
            agent.reset()
        self._pending_pretranslated_second_hops.clear()
        self.translator_pool.eval()
        for node in self.ctx.nodes:
            self.ctx.tp.get_model(node.id).eval()

        self._log_example_start(question=question, gold_answers=gold_answers, example_index=example_index)
        transcript = self.build_initial_prompt_for_agent(
            self.hub_agent,
            context,
            question,
            hub_agent_id=self.hub_agent.node_id,
            agent_count=len(self.agent_sequence),
        )
        turns: List[AgentTurnRecord] = []

        # The first node is the HubAgent and starts from native Base Context + Prompt.
        generation = self.hub_agent.generate_response(transcript)
        if self.max_turns > 1:
            next_target = self._agent_for_turn(1)
            if self.hub_agent.node_id != next_target.node_id:
                self._prepare_outgoing_route_translation(
                    source_agent=self.hub_agent,
                    logical_target_agent=next_target,
                )
        self._update_peak_memory_breakdown()
        transcript = self._append_turn_to_transcript(transcript, self.hub_agent.node_id, generation.text)
        record = self._turn_record(generation)
        turns.append(record)

        last_response = generation.text
        if self.cache_mode == CACHE_MODE_FREE:
            transcript, last_response = self._run_free_turns(
                transcript=transcript,
                initial_generation=generation,
                turns=turns,
                question=question,
                example_index=example_index,
            )
        else:
            transcript, last_response = self._run_retain_turns(
                transcript=transcript,
                initial_generation=generation,
                turns=turns,
                question=question,
                example_index=example_index,
            )

        prediction = self.extract_final_answer(transcript, last_response)
        normalized_prediction = prediction.strip().lower()
        normalized_gold = {str(answer).strip().lower() for answer in gold_answers}
        accuracy = float(normalized_prediction in normalized_gold)
        self._log_example_end(example_index=example_index, prediction=prediction, accuracy=accuracy)
        return AgentRunnerResult(
            question=question,
            gold_answers=list(gold_answers),
            prediction=prediction,
            accuracy=accuracy,
            transcript=transcript,
            turns=turns,
            profile={},
            agent_ids=list(self.node_ids),
            hub_agent_id=self.hub_agent.node_id,
            cache_mode=self.cache_mode,
        )

    def _turn_record(
        self,
        generation: AgentGeneration,
        *,
        translated_edge_id: Optional[str] = None,
        translated_offload_kind: Optional[str] = None,
        tokens_received: int = 0,
        offload_edge_id: Optional[str] = None,
        offload_kind: Optional[str] = None,
        tokens_sent: int = 0,
        tokens_sent_check_passed: bool = True,
    ) -> AgentTurnRecord:
        agent = self.agents[generation.agent_id]
        return AgentTurnRecord(
            agent_id=generation.agent_id,
            prompt=generation.prompt_text,
            response=generation.text,
            tokens_before=generation.tokens_before,
            tokens_after=generation.tokens_after,
            tokens_prompt=generation.tokens_prompt,
            tokens_completion=generation.tokens_completion,
            cache_mode=self.cache_mode,
            is_hub=agent.node_id == self.hub_agent.node_id,
            translated_edge_id=translated_edge_id,
            translated_offload_kind=translated_offload_kind,
            tokens_received=tokens_received,
            offload_edge_id=offload_edge_id,
            offload_kind=offload_kind,
            tokens_sent=tokens_sent,
            tokens_sent_check_passed=tokens_sent_check_passed,
        )

    def run(
        self,
        *,
        context: str,
        question: str,
        gold_answers: Sequence[str],
        example_index: Optional[int] = None,
    ) -> AgentRunnerResult:
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            device_obj = torch.device(self.device)
            device_index = torch.cuda.current_device() if device_obj.index is None else device_obj.index
            torch.cuda.synchronize(device_index)
        started_at = time.perf_counter()
        result = self._run_example_impl(
            context=context,
            question=question,
            gold_answers=gold_answers,
            example_index=example_index,
        )
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            device_obj = torch.device(self.device)
            device_index = torch.cuda.current_device() if device_obj.index is None else device_obj.index
            torch.cuda.synchronize(device_index)
        latency_sec = time.perf_counter() - started_at
        peak_memory = self._peak_memory_breakdown_bytes
        result.profile = {
            "latency_sec": float(latency_sec),
            "tokens": len(result.turns) * max(1, self.generation_max_new_tokens),
            "num_agent_turns": len(result.turns),
            "requested_max_turns": self.max_turns,
            "model_memory_gib": None if peak_memory is None else peak_memory.model_bytes / (1024 ** 3),
            "translator_memory_gib": None if peak_memory is None else peak_memory.translator_bytes / (1024 ** 3),
            "kv_memory_gib": None if peak_memory is None else peak_memory.kv_bytes / (1024 ** 3),
        }
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
]
