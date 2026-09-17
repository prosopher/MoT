from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import logging
import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    from json_repair import repair_json as _repair_json
except ImportError:  # requirements installs json-repair; keep source-tree tests runnable without it.
    _repair_json = None

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
OFFLOAD_KIND_NOOP = "noop"


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
    max_turns: int = 7
    generation_max_new_tokens: int = 1024
    generation_temperature: float = 1.0
    max_prompt_tokens: Optional[int] = None
    seed: int = 42
    log_agents: bool = True
    log_max_chars: int = 600
    cache_mode: str = CACHE_MODE_RETAIN
    agent_count: Optional[int] = 3


@dataclass
class AgentMessageRecord:
    agent_id: str
    prompt: str
    response: str
    tokens_before: int
    tokens_after: int
    tokens_prompt: int
    tokens_completion: int
    ttft_sec: Optional[float] = None
    cache_mode: str = CACHE_MODE_RETAIN
    is_hub: bool = False
    translated_edge_id: Optional[str] = None
    translated_offload_kind: Optional[str] = None
    tokens_received: int = 0
    offload_edge_id: Optional[str] = None
    offload_kind: Optional[str] = None
    tokens_sent: int = 0
    tokens_sent_check_passed: bool = True
    solution: Optional[str] = None
    response_state: Optional[str] = None
    memory_tokens_after: int = 0
    verification_attempts: int = 0
    verification_syntax_failures: int = 0
    verification_semantic_failures: int = 0
    verification_passed: Optional[bool] = None
    verification_reason: Optional[str] = None
    verification_syntax_passed: Optional[bool] = None
    verification_syntax_reason: Optional[str] = None
    verification_semantic_passed: Optional[bool] = None
    verification_semantic_reason: Optional[str] = None
    agreement_marker: Optional[str] = None
    final_answer: Optional[str] = None


@dataclass(frozen=True)
class StrategyQAVerificationResult:
    passed: bool
    syntax_passed: bool
    syntax_reason: str
    semantic_passed: Optional[bool]
    semantic_reason: str
    marker: Optional[str]
    final_answer: Optional[str]

    @property
    def reason(self) -> str:
        if not self.syntax_passed:
            return self.syntax_reason
        if self.semantic_passed is not True:
            return self.semantic_reason
        return "ok"


@dataclass(frozen=True)
class VerificationStageCounts:
    syntax_failures: int = 0
    semantic_failures: int = 0


@dataclass
class AgentTurnRecord:
    turn_index: int
    agent_id: str
    agent_votes: Dict[str, str]
    vote_counts: Dict[str, int]
    consensus_reached: bool
    consensus_answer: Optional[str] = None


@dataclass
class AgentRunnerResult:
    question: str
    gold_answers: List[str]
    prediction: str
    accuracy: float
    transcript: str
    agent_messages: List[AgentMessageRecord]
    turns: List[AgentTurnRecord]
    profile: Dict[str, Any]
    agent_ids: List[str]
    hub_agent_id: str
    personas: Dict[str, Tuple[str, str]]
    cache_mode: str


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
        """Translate the provided source KV span into the target model space.

        Free mode passes the full source cache. Retain mode may pass only the
        target-missing suffix. Translation itself remains algorithm-owned; this
        adapter only controls which source span is presented to it.
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
    def refresh_pretranslated_cache(
        self,
        *,
        source_agent: Agent,
        target_agent: Agent,
        prefix_tokens: int = 0,
    ) -> Dict[str, Any]:
        """Prepare translated KV for the source suffix beginning at ``prefix_tokens``.

        ``prefix_tokens=0`` preserves the original full-cache translation used by
        cache_mode=free.  Retain mode passes the target resident-prefix length so
        only the missing source KV suffix is translated before handoff.
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

        prefix_tokens = int(prefix_tokens)
        if prefix_tokens < 0 or prefix_tokens >= source_tokens:
            raise ValueError(
                f"Invalid pretranslation prefix for {source_agent.node_id}->{target_agent.node_id}: "
                f"prefix_tokens={prefix_tokens} source_tokens={source_tokens}"
            )

        translated_source_past = source_agent.past_key_values
        translated_token_ids = source_token_ids
        if prefix_tokens:
            translated_source_past = slice_past_suffix(source_agent.past_key_values, prefix_tokens)
            translated_token_ids = source_token_ids[prefix_tokens:]

        edge = self._get_edge(source_agent.node_id, target_agent.node_id)
        edge_id = edge.id
        cached_ids = source_agent.pretranslated_token_ids_by_edge.get(edge_id)
        cached_past = source_agent.pretranslated_past_by_edge.get(edge_id)
        if cached_past is not None and cached_ids == translated_token_ids:
            return {
                "edge_id": edge_id,
                "prepared_tokens": len(translated_token_ids),
                "prefix_tokens": prefix_tokens,
                "cached": True,
            }

        edge_id, translated_past = self.build_pretranslated_past_for_edge(
            source_agent=source_agent,
            target_agent=target_agent,
            source_past_key_values=translated_source_past,
            source_token_ids=translated_token_ids,
        )
        source_agent.set_pretranslated_cache(
            edge_id=edge_id,
            past_key_values=translated_past,
            cache_token_ids=translated_token_ids,
        )
        return {
            "edge_id": edge_id,
            "prepared_tokens": len(translated_token_ids),
            "prefix_tokens": prefix_tokens,
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

        # Free mode stores a full translated cache and slices it at handoff.
        # Retain mode stores only source[prefix_tokens:] because the target already
        # owns the resident prefix.  Accept exactly those two representations.
        if translated_token_ids == source_token_ids:
            translated_piece = slice_past_suffix(translated_full_past, prefix_tokens)
        elif translated_token_ids == source_token_ids[prefix_tokens:]:
            translated_piece = translated_full_past
        else:
            raise RuntimeError(
                f"Stale pretranslated cache for {edge_id}: "
                f"prepared_tokens={len(translated_token_ids)} current_tokens={len(source_token_ids)} "
                f"prefix_tokens={prefix_tokens}"
            )

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

        No translation is performed here. A prepared full cache (free) or translated
        missing suffix (retain) must already exist on source_agent; offload_cache()
        selects that delta representation and concatenates it to the target cache.
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



# MALLM options used by AgentRunner:
#   Persona Generator   = Expert
#   Response Generator  = Simple
#   Discussion Paradigm = Memory
#   Decision Protocol   = Supermajority Consensus (corrected)
#   CoT                 = OFF

MALLM_SIMPLE_PROPOSE_PROMPT = "Propose a solution."
MALLM_SIMPLE_RESPONSE_PROMPT = (
    "Improve the current solution. If you agree with the current solution, answer with [AGREE], "
    "else answer with [DISAGREE] and explain why and provide an improved solution."
)
MALLM_MEMORY_HEADER = "This is the discussion to the current point: "
MALLM_SUPERMAJORITY_THRESHOLD = 0.66

# Separate stochastic streams so changing Agent Count does not change the RNG
# state seen by otherwise-identical persona/response/verification calls. Stream
# names are part of a stable cryptographic seed derivation below; never use
# Python's process-randomized hash().
_GENERATION_SEED_STREAMS = frozenset({"persona", "discussion", "verification", "tie_break"})


MALLM_EXPERT_PERSONA_SYSTEM_PROMPT = """
When faced with a task, begin by identifying the participants who will contribute to solving the task. Provide role and description of the participants, describing their expertise or needs, formatted using the provided JSON schema.
Generate one participant at a time, complementing the existing participants to foster a rich discussion.

Example 1:
Task: Explain the basics of machine learning to high school students.
New Participant:
{"role": "Educator", "description": "An experienced teacher who simplifies complex topics for teenagers."}

Example 2:
Task: Develop a new mobile app for tracking daily exercise.
Already Generated Participants:
{"role": "Fitness Coach", "description": "A person that has high knowledge about sports and fitness."}
New Participant:
{"role": "Software Developer", "description": "A creative developer with experience in mobile applications and user interface design."}

Example 3:
Task: Write a guide on how to cook Italian food for beginners.
Already Generated Participants:
{"role": "Italian Native", "description": "An average home cook that lived in italy for 30 years."}
{"role": "Food Scientist", "description": "An educated scientist that knows which flavor combinations result in the best taste."}
New Participant:
{"role": "Chef", "description": "A professional chef specializing in Italian cuisine who enjoys teaching cooking techniques."}
        """

MALLM_EXPERT_PERSONA_FINAL_USER_PROMPT = (
    "Please use the following examples to generate a useful persona for the task! "
    "Only answer with the JSON for the next persona!"
)



class AgentRunner:
    def __init__(
        self,
        *,
        ctx: Context,
        translator_pool,
        alg: str,
        max_turns: int = 7,
        generation_max_new_tokens: int = 1024,
        generation_temperature: float = 1.0,
        max_prompt_tokens: Optional[int] = None,
        seed: int = 42,
        log_agents: bool = True,
        log_max_chars: int = 600,
        cache_mode: str = CACHE_MODE_RETAIN,
        agent_count: Optional[int] = 3,
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
        self.generation_temperature = float(generation_temperature)
        if self.generation_temperature < 0.0:
            raise ValueError(
                f"generation_temperature must be >= 0, got {self.generation_temperature}"
            )
        self.max_prompt_tokens = max_prompt_tokens
        self.seed = int(seed)
        self.log_agents = bool(log_agents)
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
        # Stop behavior must be Agent-Count invariant.  Using one stop string per
        # active agent made the exact same completion terminate differently when
        # scaling, e.g. "\nAgent C:" was a stop only when C existed.  A generic
        # speaker marker catches every logical Agent without depending on count.
        stop_sequences = (
            "\nAgent ",
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
                    temperature=self.generation_temperature,
                )
            )

        self.hub_agent: HubAgent = self.agent_sequence[0]  # type: ignore[assignment]
        self.non_hub_agents = self.agent_sequence[1:]
        self.agents = {agent.node_id: agent for agent in self.agent_sequence}
        self.agent_personas: Dict[str, Tuple[str, str]] = {}
        self._consensus_reached = False
        self._consensus_turn: Optional[int] = None
        self._consensus_answer: Optional[str] = None
        self._final_agent_votes: Dict[str, str] = {}
        self._final_decision_method: Optional[str] = None
        self._final_decision_answer: Optional[str] = None
        self._current_example_seed = self.seed

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
        # Actual live KV/cache tensor storage, deduplicated by underlying storage.
        # This is the authoritative KV-cache memory metric; unlike allocator
        # residuals it does not include attention activations or CUDA workspaces.
        self._peak_kv_cache_bytes = 0
        self._kv_cache_memory_samples_bytes: List[int] = []

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
            generation_temperature=config.generation_temperature,
            max_prompt_tokens=config.max_prompt_tokens,
            seed=config.seed,
            log_agents=config.log_agents,
            log_max_chars=config.log_max_chars,
            cache_mode=config.cache_mode,
            agent_count=config.agent_count,
        )

    def _set_generation_seed(
        self,
        stream: str,
        *,
        turn_index: int = 0,
        agent_index: int = 0,
        attempt: int = 0,
    ) -> int:
        """Seed one logical generation independently of Agent Count.

        The seed is derived only from experiment-invariant coordinates:
        base/example seed, stream, logical turn, Agent offset, and retry index.
        Agent Count is deliberately absent.  A stable BLAKE2 digest avoids the
        arithmetic collisions possible with ``turn*k + agent*m + attempt`` and
        remains identical across Python processes/machines.
        """
        if stream not in _GENERATION_SEED_STREAMS:
            raise ValueError(f"Unknown generation seed stream: {stream!r}")
        turn_index = max(0, int(turn_index))
        agent_index = max(0, int(agent_index))
        attempt = max(0, int(attempt))
        material = (
            f"agent-runner-v2|base={int(self.seed)}|example={int(self._current_example_seed)}|"
            f"stream={stream}|turn={turn_index}|agent={agent_index}|attempt={attempt}"
        ).encode("utf-8")
        digest = hashlib.blake2b(material, digest_size=8, person=b"MALLMSeed").digest()
        seed = int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)
        set_seed(seed)
        return seed

    @staticmethod
    def _persona_is_usable(
        persona: Tuple[str, str],
        existing_personas: Sequence[Tuple[str, str]],
    ) -> bool:
        """Reject duplicate or obviously corrupted Expert personas before use."""
        role, description = persona
        normalized_role = re.sub(r"\s+", " ", role).strip().casefold()
        if any(
            normalized_role == re.sub(r"\s+", " ", old_role).strip().casefold()
            for old_role, _ in existing_personas
        ):
            return False
        if "```" in role or "```" in description or "<|" in role or "<|" in description:
            return False
        # The Expert prompt is English. Non-ASCII/control artifacts in generated
        # personas are a strong signal of corrupted sampling and were observed in
        # larger-agent logs, where bad late personas can degrade the discussion.
        if not role.isascii() or not description.isascii():
            return False
        if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in role + description):
            return False
        if len(role) > 120 or len(description) > 800:
            return False
        return True

    @staticmethod
    def _task_instruction() -> str:
        # StrategyQA uses only the semantic Yes/No answer space. Agreement markers
        # remain discussion-control signals and are intentionally separate.
        return "Decide whether the answer to the following question is Yes or No. When you propose or revise a solution, make the final Yes/No conclusion explicit."

    @staticmethod
    def _task_instruction_with_context(context: str) -> str:
        instruction = AgentRunner._task_instruction()
        if context.strip():
            return f"{instruction}\nContext:\n{context.strip()}"
        return instruction

    @staticmethod
    def _default_persona(index: int = 0) -> Tuple[str, str]:
        return (f"Participant {index + 1}", "Contribute a useful and complementary perspective to the task.")

    @staticmethod
    def _role_text(persona: Tuple[str, str]) -> str:
        role, description = persona
        return f"{role} ({description})"

    @staticmethod
    def _memory_speaker_text(persona: Tuple[str, str], response: str) -> str:
        role, _ = persona
        return f"{role}: {(response or '').strip()}"

    @staticmethod
    def _try_parse_persona(text: str) -> Optional[Tuple[str, str]]:
        raw = (text or "").strip()
        payload: Any = None

        # ExpertGenerator in the official implementation repairs JSON before
        # decoding and accepts a list by taking its first element. Use the same
        # library when installed; retain a strict/fallback parser for source-tree
        # test environments that have not installed requirements yet.
        if _repair_json is not None:
            try:
                payload = json.loads(_repair_json(raw))
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = None
        if payload is None:
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
                if match is not None:
                    try:
                        payload = json.loads(match.group(0))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        payload = None

        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if not isinstance(payload, dict):
            return None
        role = str(payload.get("role", "")).strip()
        description = str(payload.get("description", "")).strip()
        if role and description:
            return role, description
        return None

    @staticmethod
    def _parse_persona(text: str, index: int) -> Tuple[str, str]:
        persona = AgentRunner._try_parse_persona(text)
        if persona is not None:
            return persona
        logging.warning("Could not parse Expert persona JSON; using Participant %d fallback.", index + 1)
        return AgentRunner._default_persona(index)

    def _build_expert_persona_prompt(
        self,
        *,
        context: str,
        question: str,
        existing_personas: Sequence[Tuple[str, str]],
    ) -> str:
        # Match ExpertGenerator.generate_persona from the official MALLM code.
        # The coordinator passes task_instruction (with Context appended) plus
        # input_str as task_description. Existing personas are a separate SYSTEM
        # message, followed by a final USER instruction.
        task_description = f"{self._task_instruction_with_context(context)} {question.strip()}"
        messages = [
            {"role": "system", "content": MALLM_EXPERT_PERSONA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "\nNow generate a participant to discuss the following task:\n"
                    f"Task: {task_description}\n"
                ),
            },
        ]
        if existing_personas:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Already Generated Participants:\n"
                        + "\n".join(
                            str({"role": role, "description": description})
                            for role, description in existing_personas
                        )
                    ),
                }
            )
        messages.append(
            {"role": "user", "content": MALLM_EXPERT_PERSONA_FINAL_USER_PROMPT}
        )

        tokenizer = self.hub_agent.model.tokenizer
        template = self._explicit_chat_template(self.hub_agent)
        if template is not None:
            # Prefer the exact official message topology. If a checkpoint chat
            # template cannot represent SYSTEM messages, fold the same contents
            # into a USER message only as a tokenizer-compatibility fallback.
            fallback_user = "\n\n".join(str(message["content"]) for message in messages)
            for candidate_messages in (
                messages,
                [{"role": "user", "content": fallback_user}],
            ):
                try:
                    rendered = tokenizer.apply_chat_template(
                        candidate_messages,
                        chat_template=template,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if rendered:
                        return self._strip_irrelevant_chat_template_metadata(str(rendered))
                except Exception as error:
                    logging.debug("Expert persona chat-template rendering failed: %s", error)
        return "\n\n".join(str(message["content"]) for message in messages) + "\n"

    def _generate_expert_personas(self, context: str, question: str) -> Dict[str, Tuple[str, str]]:
        personas: Dict[str, Tuple[str, str]] = {}
        existing: List[Tuple[str, str]] = []
        for index, agent in enumerate(self.agent_sequence):
            prompt = self._build_expert_persona_prompt(
                context=context,
                question=question,
                existing_personas=existing,
            )
            persona: Optional[Tuple[str, str]] = None
            for attempt in range(5):
                self._set_generation_seed(
                    "persona",
                    agent_index=index,
                    attempt=attempt,
                )
                generator = Agent(
                    node_id=f"persona-{index}",
                    model=agent.model,
                    device=self.device,
                    max_new_tokens=self.generation_max_new_tokens,
                    max_prompt_tokens=self.max_prompt_tokens,
                    temperature=self.generation_temperature,
                )
                generation = generator.generate_response(prompt)
                generator.reset()
                candidate = self._try_parse_persona(generation.text)
                if candidate is not None and self._persona_is_usable(candidate, existing):
                    persona = candidate
                    break
                if candidate is not None:
                    logging.warning(
                        "Rejecting duplicate/corrupted Expert persona at index %d: %r",
                        index,
                        candidate,
                    )
            if persona is None:
                logging.warning("Could not parse Expert persona JSON after 5 attempts; using Participant %d fallback.", index + 1)
                persona = self._default_persona(index)
            personas[agent.node_id] = persona
            existing.append(persona)
        return personas

    @staticmethod
    def _general_debate_prompt(
        context: str,
        question: str,
        *,
        persona: Tuple[str, str],
        current_solution: str = "",
    ) -> str:
        # Match SimpleResponseGenerator.get_filled_template. In the official
        # coordinator, context is appended to task_instruction; input_str itself
        # remains the task input. The Memory appendix is kept in the reusable KV
        # prefix in this implementation, so it is intentionally omitted here.
        base = (
            f"{AgentRunner._system_prompt()}\n"
            f"Task: {AgentRunner._task_instruction_with_context(context)}\n"
            f"Input: {question.strip()}\n"
            f"Your role: {AgentRunner._role_text(persona)}\n"
            "Response format: end every non-bare response with exactly "
            "`Final Solution: Yes` or `Final Solution: No`."
        )
        if current_solution.strip():
            base += f"\nCurrent Solution: {current_solution.strip()}"
        return base

    @staticmethod
    def _build_plain_initial_prompt(
        context: str,
        question: str,
        *,
        persona: Tuple[str, str] = ("Participant 1", "Contribute a useful and complementary perspective to the task."),
    ) -> str:
        return (
            AgentRunner._general_debate_prompt(context, question, persona=persona)
            + "\n\n"
            + MALLM_SIMPLE_PROPOSE_PROMPT
            + "\n### Response:\n"
        )

    @staticmethod
    def _build_plain_followup_prompt(
        question: str,
        *,
        context: str = "",
        persona: Tuple[str, str] = ("Participant 1", "Contribute a useful and complementary perspective to the task."),
        current_solution: str = "",
    ) -> str:
        return (
            "\n"
            + AgentRunner._general_debate_prompt(
                context,
                question,
                persona=persona,
                current_solution=current_solution,
            )
            + "\n\n"
            + MALLM_SIMPLE_RESPONSE_PROMPT
            + "\n### Response:\n"
        )

    @staticmethod
    def _system_prompt() -> str:
        return "You take part in a discussion to solve a task."

    @staticmethod
    def _build_initial_user_content(
        context: str,
        question: str,
        *,
        persona: Tuple[str, str] = ("Participant 1", "Contribute a useful and complementary perspective to the task."),
    ) -> str:
        return AgentRunner._general_debate_prompt(context, question, persona=persona)

    @staticmethod
    def _build_followup_user_content(
        question: str,
        *,
        context: str = "",
        persona: Tuple[str, str] = ("Participant 1", "Contribute a useful and complementary perspective to the task."),
        current_solution: str = "",
    ) -> str:
        # Plain-text/debug representation of the official Simple SYSTEM + USER
        # prompt topology used by this four-component configuration.
        return (
            AgentRunner._general_debate_prompt(
                context,
                question,
                persona=persona,
                current_solution=current_solution,
            )
            + "\n\n"
            + MALLM_SIMPLE_RESPONSE_PROMPT
        )

    @staticmethod
    def _normalize_strategyqa_answer(answer: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", str(answer or "").strip())
        if not text:
            return None
        compact = re.sub(r"[^a-z0-9]", "", text.lower())
        aliases = {
            "yes": "Yes",
            "no": "No",
        }
        return aliases.get(compact)

    @staticmethod
    def _explicit_strategyqa_assertions(text: str) -> List[Tuple[int, str]]:
        """Return explicit Yes/No assertions, excluding ambiguous or rejected mentions.

        This is intentionally conservative.  It catches ordinary assertions such
        as ``"Yes", because ...`` or ``No Explanation: ...`` that are not strong
        enough to be a labelled final-answer pattern, while ignoring phrases like
        ``Yes or No`` and ``Yes is incorrect``.  The verifier uses these assertions
        to prevent a long [AGREE] response from silently arguing for the opposite
        semantic answer.
        """
        raw = str(text or "")
        assertions: List[Tuple[int, str]] = []
        pattern = re.compile(
            r"(?<![A-Za-z0-9])[\"'`*]*(Yes|No)[\"'`*]*"
            r"(?=\s*(?:[,.;:!?)]|$|\n|(?:Explanation|Reasoning)\s*:))",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(raw):
            value = AgentRunner._normalize_strategyqa_answer(match.group(1))
            if value is None:
                continue
            before = raw[max(0, match.start() - 40) : match.start()].lower()
            after = raw[match.end() : match.end() + 64].lower()

            # ``Yes or No`` / ``No or Yes`` names the answer space, not a claim.
            if re.match(r"\s*(?:or|/)\s*(?:yes|no)\b", after):
                continue
            if re.search(r"(?:yes|no)\s*(?:or|/)\s*$", before):
                continue

            # Reject mentions that explicitly say the token is wrong/incorrect.
            if re.match(
                r"\s*(?:is|would\s+be|seems|appears)?\s*"
                r"(?:the\s+)?(?:incorrect|wrong|false|not\s+(?:the\s+)?correct)\b",
                after,
            ):
                continue
            if re.search(r"\b(?:not|isn't|isnt|is\s+not)\s*$", before):
                continue

            assertions.append((match.start(), value))
        return assertions

    @staticmethod
    def _parse_reasoning_stance(text: str) -> Optional[str]:
        """Parse the stage-2 reasoning stance classifier's closed output."""
        raw = str(text or "").strip()
        if not raw:
            return None
        first_line = raw.splitlines()[0].strip()
        first_line = first_line.strip("`*_\'\" \t\r\n.!,:;").upper()
        mapping = {
            "SUPPORTS_YES": "Yes",
            "SUPPORTS_NO": "No",
            "UNCLEAR": "unclear",
            "CONTRADICTORY": "contradictory",
        }
        return mapping.get(first_line)

    @staticmethod
    def _parse_improvement_verdict(text: str) -> Optional[bool]:
        """Parse the same-answer DISAGREE improvement verifier's closed output."""
        raw = str(text or "").strip()
        if not raw:
            return None
        first_line = raw.splitlines()[0].strip()
        first_line = first_line.strip("`*_\'\" \t\r\n.!,:;").upper()
        if first_line == "SUBSTANTIVE":
            return True
        if first_line == "NOT_SUBSTANTIVE":
            return False
        return None

    @staticmethod
    def _extract_strategyqa_answer(text: str) -> Optional[str]:
        """Extract the concluded StrategyQA answer using only Yes/No semantics.

        Discussion reasoning can mention both truth values while rejecting one of
        them, so conclusion-oriented statements and later statements take priority.
        Ambiguous answer-space phrases such as ``Yes or No`` are never accepted as
        a concrete answer.
        """
        text = str(text or "").strip()
        if not text:
            return None

        candidates: List[Tuple[int, int, str]] = []

        def add_candidate(position: int, priority: int, value: str) -> None:
            normalized = AgentRunner._normalize_strategyqa_answer(value)
            if normalized is not None:
                candidates.append((position, priority, normalized))

        def is_ambiguous_suffix(end_position: int) -> bool:
            suffix = text[end_position : end_position + 32]
            return re.match(r"\s*(?:or|/)\s*(?:Yes|No)\b", suffix, flags=re.IGNORECASE) is not None

        # Strong explicit conclusion labels used by answer parsing/verification.
        labelled = re.compile(
            r"(?:final\s+(?:solution|answer)|improved\s+solution|proposed\s+solution|"
            r"(?:the\s+)?answer|(?:the\s+)?solution|conclusion)"
            r"\s*(?:is|:|=)?\s*\**\s*(Yes|No)\b",
            flags=re.IGNORECASE,
        )
        for match in labelled.finditer(text):
            if not is_ambiguous_suffix(match.end(1)):
                add_candidate(match.start(), 5, match.group(1))

        # Common discourse conclusions.
        discourse = re.compile(
            r"(?:therefore|thus|so|hence)\s*,?\s*(?:the\s+(?:answer|conclusion)\s+(?:is|:)?\s*)?"
            r"(Yes|No)\b",
            flags=re.IGNORECASE,
        )
        for match in discourse.finditer(text):
            if not is_ambiguous_suffix(match.end(1)):
                add_candidate(match.start(), 4, match.group(1))

        # A line beginning with Yes/No is a useful signal for normal first
        # proposals such as 'Yes, because ...'.
        line_lead = re.compile(r"(?m)^\s*(Yes|No)\s*(?:[.!,:;\-]|$)", flags=re.IGNORECASE)
        for match in line_lead.finditer(text):
            suffix = text[match.end() : match.end() + 64].lower()
            rejected = re.match(
                r"\s*(?:is|would\s+be|seems|appears)?\s*(?:the\s+)?"
                r"(?:incorrect|wrong|false|not\s+(?:the\s+)?correct)\b",
                suffix,
            )
            if not rejected:
                add_candidate(match.start(), 3, match.group(1))

        # Weak explicit assertions catch forms such as '"Yes", because ...' that
        # otherwise caused [AGREE] verification to miss an opposite answer.
        for position, answer in AgentRunner._explicit_strategyqa_assertions(text):
            add_candidate(position, 2, answer)

        if candidates:
            return max(candidates, key=lambda item: (item[0], item[1]))[2]

        # A response may consist of just the closed-set token.
        compact = re.sub(r"^[\s\[\]`*_:.-]+|[\s\[\]`*_.:,;!\-]+$", "", text)
        return AgentRunner._normalize_strategyqa_answer(compact)

    @staticmethod
    def _extract_agreement_marker(response: str) -> Optional[str]:
        """Return Simple's control marker only when it is the first response token.

        The official Simple prompt asks for the literal ``[AGREE]`` / ``[DISAGREE]``
        control token. Treating arbitrary prose such as ``I disagree`` or a later
        mention of the word "agree" as a control signal makes vote parsing depend on
        incidental reasoning text, so only the bracketed first-line marker is valid.
        """
        text = str(response or "").lstrip()
        match = re.match(r"^\[\s*(AGREE|DISAGREE)\s*\]", text, flags=re.IGNORECASE)
        if match is None:
            return None
        return match.group(1).lower()

    @staticmethod
    def _extract_final_solution_line(response: str) -> Optional[str]:
        """Return the exact Yes/No claim from the final non-empty line.

        Verification intentionally does not infer the vote from arbitrary prose.
        The final syntactic contract is a terminal ``Final Solution: Yes|No`` line.
        """
        lines = [line.strip() for line in str(response or "").splitlines() if line.strip()]
        if not lines:
            return None
        match = re.fullmatch(
            r"Final\s+Solution\s*:\s*(Yes|No)\s*[.!]?",
            lines[-1],
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        return AgentRunner._normalize_strategyqa_answer(match.group(1))

    @staticmethod
    def _verify_initial_syntax(response: str) -> StrategyQAVerificationResult:
        """Stage 1 for the first proposal: deterministic syntax only."""
        if AgentRunner._extract_agreement_marker(response) is not None:
            return StrategyQAVerificationResult(
                passed=False,
                syntax_passed=False,
                syntax_reason="initial proposal must not start with [AGREE]/[DISAGREE]",
                semantic_passed=None,
                semantic_reason="not run",
                marker=None,
                final_answer=AgentRunner._extract_final_solution_line(response),
            )
        final_answer = AgentRunner._extract_final_solution_line(response)
        if final_answer is None:
            return StrategyQAVerificationResult(
                passed=False,
                syntax_passed=False,
                syntax_reason=(
                    "initial proposal must end with exactly `Final Solution: Yes` "
                    "or `Final Solution: No`"
                ),
                semantic_passed=None,
                semantic_reason="not run",
                marker=None,
                final_answer=None,
            )
        return StrategyQAVerificationResult(
            passed=False,
            syntax_passed=True,
            syntax_reason="ok",
            semantic_passed=None,
            semantic_reason="not run",
            marker=None,
            final_answer=final_answer,
        )

    @staticmethod
    def _verify_followup_syntax(response: str) -> StrategyQAVerificationResult:
        """Stage 1 for follow-ups: deterministic marker/answer structure only."""
        marker = AgentRunner._extract_agreement_marker(response)
        if marker is None:
            return StrategyQAVerificationResult(
                passed=False,
                syntax_passed=False,
                syntax_reason="missing first-token [AGREE]/[DISAGREE] marker",
                semantic_passed=None,
                semantic_reason="not run",
                marker=None,
                final_answer=AgentRunner._extract_final_solution_line(response),
            )
        # Syntax is deliberately closed: prose-level Yes/No inference belongs to
        # semantic verification, not to this deterministic stage. A bare [AGREE]
        # is the only response that may omit the terminal Final Solution line.
        stripped = str(response or "").strip()
        bare_agree = marker == "agree" and re.fullmatch(
            r"\[\s*AGREE\s*\]", stripped, flags=re.IGNORECASE
        ) is not None
        final_answer = AgentRunner._extract_final_solution_line(response)
        if final_answer is None and not bare_agree:
            return StrategyQAVerificationResult(
                passed=False,
                syntax_passed=False,
                syntax_reason=(
                    "follow-up must end with exactly `Final Solution: Yes` or "
                    "`Final Solution: No` (except bare [AGREE])"
                ),
                semantic_passed=None,
                semantic_reason="not run",
                marker=marker,
                final_answer=None,
            )
        return StrategyQAVerificationResult(
            passed=False,
            syntax_passed=True,
            syntax_reason="ok",
            semantic_passed=None,
            semantic_reason="not run",
            marker=marker,
            final_answer=final_answer,
        )

    @staticmethod
    def _solution_from_response(response: str) -> Tuple[str, Optional[str]]:
        answer = AgentRunner._extract_strategyqa_answer(response)
        if answer is not None:
            return answer, answer
        return (response or "").strip(), None

    @staticmethod
    def _semantic_reasoning_body(response: str) -> str:
        """Remove control/final-answer syntax before stage-2 reasoning checks.

        Stage 2 receives marker/final_answer as separately parsed facts. Keeping
        those literal tokens inside the prose biases a verifier toward parroting
        the terminal label instead of checking whether the reasoning supports it.
        """
        text = str(response or "").strip()
        text = re.sub(
            r"^\s*\[\s*(?:AGREE|DISAGREE)\s*\]\s*",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        lines = [line.rstrip() for line in text.splitlines()]
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and re.fullmatch(
            r"\s*Final\s+Solution\s*:\s*(?:Yes|No)\s*[.!]?\s*",
            lines[-1],
            flags=re.IGNORECASE,
        ):
            lines.pop()
        return "\n".join(lines).strip()

    def _render_verifier_messages(self, agent: Agent, messages: List[Dict[str, str]]) -> str:
        tokenizer = agent.model.tokenizer
        template = self._explicit_chat_template(agent)
        if template is not None:
            fallback_user = "\n\n".join(str(message["content"]) for message in messages)
            for candidate_messages in (
                messages,
                [{"role": "user", "content": fallback_user}],
            ):
                try:
                    rendered = tokenizer.apply_chat_template(
                        candidate_messages,
                        chat_template=template,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if rendered:
                        return self._strip_irrelevant_chat_template_metadata(str(rendered))
                except Exception as error:
                    logging.debug("Semantic-verification chat-template rendering failed: %s", error)
        return "\n\n".join(str(message["content"]) for message in messages) + "\n"

    def _build_reasoning_stance_prompt(
        self,
        *,
        agent: Agent,
        context: str,
        question: str,
        response: str,
    ) -> str:
        """Ask what answer the reasoning itself supports, without exposing labels.

        The parsed marker, current answer, and terminal Final Solution are deliberately
        absent from this prompt.  This prevents the semantic model from anchoring on
        the claimed label while still letting the caller compare the independently
        inferred reasoning stance against those syntactically parsed fields.
        """
        reasoning_body = self._semantic_reasoning_body(response) or "[no reasoning body]"
        messages = [
            {
                "role": "system",
                "content": (
                    "You classify the stance expressed by a response's reasoning. "
                    "Do not decide what the objectively correct answer should be. "
                    "Infer only what conclusion the author's own reasoning supports."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {self._task_instruction_with_context(context)}\n"
                    f"Input: {question.strip()}\n"
                    "Reasoning body (control marker and terminal Final Solution removed):\n"
                    f"{reasoning_body}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return exactly one token:\n"
                    "`SUPPORTS_YES` - the author's reasoning supports answering Yes.\n"
                    "`SUPPORTS_NO` - the author's reasoning supports answering No.\n"
                    "`UNCLEAR` - the reasoning does not establish either answer.\n"
                    "`CONTRADICTORY` - the author's own reasoning materially supports both answers or conflicts with itself.\n"
                    "Do not use the task's factual truth to override what the reasoning itself says. "
                    "Do not output any explanation."
                ),
            },
        ]
        return self._render_verifier_messages(agent, messages)

    def _build_reasoning_improvement_prompt(
        self,
        *,
        agent: Agent,
        context: str,
        question: str,
        response: str,
        current_response: Optional[str],
    ) -> str:
        """Check whether same-answer DISAGREE makes a substantive reasoning change.

        No Yes/No label or control marker is exposed here.  The model compares only
        the accepted and proposed reasoning bodies, so it cannot satisfy DISAGREE by
        parroting a desired final label.
        """
        previous_body = (
            self._semantic_reasoning_body(current_response) if current_response else ""
        ) or "[no previous reasoning body]"
        proposed_body = self._semantic_reasoning_body(response) or "[no proposed reasoning body]"
        messages = [
            {
                "role": "system",
                "content": (
                    "You compare two reasoning passages for substantive revision. "
                    "Do not judge which answer is factually correct and do not infer a Yes/No label."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {self._task_instruction_with_context(context)}\n"
                    f"Input: {question.strip()}\n"
                    f"Previously accepted reasoning:\n{previous_body}\n\n"
                    f"Proposed reasoning:\n{proposed_body}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return exactly one token: `SUBSTANTIVE` or `NOT_SUBSTANTIVE`.\n"
                    "SUBSTANTIVE means the proposed reasoning adds, corrects, replaces, or materially sharpens an argument/evidence point.\n"
                    "NOT_SUBSTANTIVE means it only rephrases, reformats, shortens, expands cosmetically, or repeats the previous reasoning.\n"
                    "Do not output any explanation."
                ),
            },
        ]
        return self._render_verifier_messages(agent, messages)

    def _run_semantic_verifier_prompt(
        self,
        *,
        agent: Agent,
        prompt: str,
    ) -> str:
        verifier = Agent(
            node_id=f"verifier-{agent.node_id}",
            model=agent.model,
            device=self.device,
            max_new_tokens=self.generation_max_new_tokens,
            max_prompt_tokens=self.max_prompt_tokens,
            temperature=0.0,
        )
        verdict = verifier.generate_response(prompt)
        self._update_peak_memory_breakdown()
        verifier.reset()
        return verdict.text.strip()

    def _verify_response_semantics_with_model(
        self,
        *,
        agent: Agent,
        context: str,
        question: str,
        response: str,
        marker: Optional[str],
        final_answer: str,
        current_answer: Optional[str] = None,
        current_response: Optional[str] = None,
    ) -> Tuple[str, Optional[bool]]:
        """Stage-2 semantic verification without final-label anchoring.

        First infer the stance supported by the reasoning body *without* exposing
        marker/current/final-answer fields.  Then compare that independent stance
        deterministically with the syntactically parsed final answer and marker.
        Only same-answer DISAGREE needs a second model check, which compares old vs
        new reasoning for substantive improvement without exposing either label.
        """
        # Bare [AGREE] intentionally carries no reasoning body; its semantic meaning
        # is fully determined by the marker plus the inherited current answer.
        reasoning_body = self._semantic_reasoning_body(response)
        if marker == "agree" and not reasoning_body:
            if current_answer in {"Yes", "No"} and final_answer == current_answer:
                return "bare_agree", True
            return "bare_agree_mismatch", False

        stance_prompt = self._build_reasoning_stance_prompt(
            agent=agent,
            context=context,
            question=question,
            response=response,
        )
        raw_stance = self._run_semantic_verifier_prompt(agent=agent, prompt=stance_prompt)
        stance = self._parse_reasoning_stance(raw_stance)
        if stance not in {"Yes", "No"}:
            return f"stance={raw_stance or 'empty'}", False
        if stance != final_answer:
            return f"stance={stance}; final={final_answer}", False

        if marker == "agree":
            if current_answer not in {"Yes", "No"} or final_answer != current_answer:
                return f"stance={stance}; agree_current={current_answer}; final={final_answer}", False
            return f"stance={stance}; agree_consistent", True

        if marker == "disagree":
            if current_answer not in {"Yes", "No"}:
                return f"stance={stance}; missing_current", False
            if final_answer != current_answer:
                # A changed conclusion is already a substantive disagreement once
                # the reasoning independently supports the new final answer.
                return f"stance={stance}; changed_answer", True

            improvement_prompt = self._build_reasoning_improvement_prompt(
                agent=agent,
                context=context,
                question=question,
                response=response,
                current_response=current_response,
            )
            raw_improvement = self._run_semantic_verifier_prompt(
                agent=agent,
                prompt=improvement_prompt,
            )
            improvement = self._parse_improvement_verdict(raw_improvement)
            return (
                f"stance={stance}; improvement={raw_improvement or 'empty'}",
                improvement is True,
            )

        # INITIAL proposal: the independently inferred reasoning stance only needs
        # to agree with the syntactically parsed final answer.
        if marker is None:
            return f"stance={stance}; initial_consistent", True

        return f"stance={stance}; unknown_marker={marker}", False

    def _supermajority_consensus(
        self,
        agent_votes: Dict[str, str],
    ) -> Optional[str]:
        """Return a >66% supermajority over each Agent's latest semantic vote.

        MALLM's ``SupermajorityConsensus`` uses a 0.66 threshold. AgentRunner
        keeps one latest verified Yes/No stance per Agent. Once consensus evaluation
        is enabled, the fraction is measured against the full configured Agent count
        and must be *strictly greater* than 0.66. The orchestration layer intentionally
        defers the first consensus evaluation until every configured Agent has completed
        at least one verified turn. Only verified messages are committed, because invalid
        generations are retried until they pass before the Agent's turn can complete.
        """
        valid_votes = {
            agent_id: answer
            for agent_id, answer in agent_votes.items()
            if answer in {"Yes", "No"}
        }
        total_agents = len(self.agent_sequence)
        if total_agents <= 0:
            return None
        yes_votes = sum(answer == "Yes" for answer in valid_votes.values())
        no_votes = sum(answer == "No" for answer in valid_votes.values())
        if yes_votes / total_agents > MALLM_SUPERMAJORITY_THRESHOLD:
            return "Yes"
        if no_votes / total_agents > MALLM_SUPERMAJORITY_THRESHOLD:
            return "No"
        return None

    def _all_agents_have_participated(
        self,
        agent_votes: Dict[str, str],
    ) -> bool:
        """Return True only after every configured Agent has a verified vote."""
        return all(agent.node_id in agent_votes for agent in self.agent_sequence)

    def _majority_vote_with_random_tie(
        self,
        agent_votes: Dict[str, str],
    ) -> Tuple[str, str]:
        """Select the final answer after max_turns when no supermajority exists.

        Each Agent contributes its latest verified Yes/No vote. A unique plurality
        winner is returned as the majority-vote result. If the highest vote count
        is tied, choose uniformly from the tied answers using the experiment's
        deterministic seed stream so repeated runs remain reproducible.
        """
        valid_votes = [answer for answer in agent_votes.values() if answer in {"Yes", "No"}]
        if not valid_votes:
            raise RuntimeError("Cannot select a final answer: no verified agent votes are available")

        vote_counts = {
            "Yes": sum(answer == "Yes" for answer in valid_votes),
            "No": sum(answer == "No" for answer in valid_votes),
        }
        max_count = max(vote_counts.values())
        winners = [answer for answer, count in vote_counts.items() if count == max_count]
        if len(winners) == 1:
            return winners[0], "majority_vote"

        self._set_generation_seed(
            "tie_break",
            turn_index=self.max_turns,
            agent_index=0,
            attempt=0,
        )
        return random.choice(winners), "random_tie_break"

    def _evaluate_turn_consensus(
        self,
        *,
        turn_index: int,
        agent_id: str,
        agent_votes: Dict[str, str],
        turns: List[AgentTurnRecord],
    ) -> Optional[str]:
        """Record a verified Turn and evaluate consensus once all Agents participated.

        The first pass through the Agent sequence is a mandatory participation phase:
        no consensus decision is attempted until every configured Agent has contributed
        at least one verified Yes/No vote. Starting with the Turn that completes that
        first pass, >66% Supermajority Consensus is evaluated after every verified Turn.
        """
        consensus_eligible = self._all_agents_have_participated(agent_votes)
        consensus_answer = (
            self._supermajority_consensus(agent_votes) if consensus_eligible else None
        )
        vote_counts = {
            "Yes": sum(answer == "Yes" for answer in agent_votes.values()),
            "No": sum(answer == "No" for answer in agent_votes.values()),
        }
        turns.append(
            AgentTurnRecord(
                turn_index=turn_index,
                agent_id=agent_id,
                agent_votes=dict(agent_votes),
                vote_counts=vote_counts,
                consensus_reached=consensus_answer is not None,
                consensus_answer=consensus_answer,
            )
        )
        self._consensus_reached = consensus_answer is not None
        self._consensus_turn = turn_index if consensus_answer is not None else None
        self._consensus_answer = consensus_answer
        self._final_agent_votes = dict(agent_votes)
        if self.log_agents:
            logging.info(
                "[AgentRunner][Turn %d/%d][Agent %s] consensus | eligible=%s | votes=%s | counts=%s | answer=%s",
                turn_index,
                self.max_turns,
                agent_id,
                consensus_eligible,
                dict(agent_votes),
                vote_counts,
                consensus_answer or "none",
            )
        return consensus_answer

    @staticmethod
    def _response_updates_solution(
        response: str,
        *,
        current_solution: str,
        current_answer: Optional[str],
        extracted_solution: Optional[str] = None,
    ) -> Tuple[str, Optional[str], str]:
        # Match ResponseGenerator.extract_agreement: agreement is true iff the
        # response contains "agree" but not "disagree" (case-insensitive).
        text = response or ""
        marker = AgentRunner._extract_agreement_marker(text)
        if marker == "agree":
            return current_solution, current_answer, "agree"

        # Agent.improve stores response.solution whenever agreement is False.
        # Preserve that extracted solution even if the StrategyQA answer parser
        # cannot normalize it; the decision protocol operates on the solution
        # string, not on task-specific evaluation metadata.
        proposed_solution = extracted_solution if extracted_solution is not None else text.strip()
        response_answer = AgentRunner._extract_strategyqa_answer(proposed_solution)
        return proposed_solution, response_answer, "revise"

    def _render_memory_entry(
        self,
        *,
        agent: Agent,
        persona: Tuple[str, str],
        response: str,
        context: str,
        question: str,
        include_base: bool,
    ) -> str:
        memory_content = self._memory_speaker_text(persona, response)
        del context, question

        # SimpleResponseGenerator appends the exact header below to its SYSTEM
        # prompt when memory is present, then appends agent-memory messages. In
        # Agent.get_discussion_history, another agent's agent message is a USER
        # message prefixed by the persona; the speaking agent's own agent message
        # is ASSISTANT. A single translated/reused KV prefix cannot be both roles
        # for different next agents, so the shared-KV representation canonicalizes
        # every agent message as the official non-self USER form. Only this
        # target-specific self-role distinction and causal placement are sacrificed.
        messages = []
        if include_base:
            messages.append({"role": "system", "content": MALLM_MEMORY_HEADER})
        messages.append({"role": "user", "content": memory_content})

        tokenizer = agent.model.tokenizer
        template = self._explicit_chat_template(agent)
        if template is not None:
            fallback_user = "\n".join(str(message["content"]) for message in messages)
            for candidate_messages in (
                messages,
                [{"role": "user", "content": fallback_user}],
            ):
                try:
                    rendered = tokenizer.apply_chat_template(
                        candidate_messages,
                        chat_template=template,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    if rendered:
                        rendered = self._strip_irrelevant_chat_template_metadata(str(rendered))
                        if not include_base:
                            rendered = self._strip_leading_bos_for_continuation(tokenizer, rendered)
                        return rendered
                except Exception as error:
                    logging.debug("Memory-entry chat-template rendering failed: %s", error)
        if include_base:
            return f"{MALLM_MEMORY_HEADER}\n{memory_content}\n"
        return f"\n{memory_content}\n"

    def _commit_discussion_memory(
        self,
        *,
        agent: Agent,
        generation: AgentGeneration,
        persona: Tuple[str, str],
        context: str,
        question: str,
    ) -> int:
        # A generation temporarily appends its control prompt and assistant answer
        # to KV. MALLM Memory stores discussion messages separately from those
        # prompts. Restore the shared-memory prefix, then append only a persona-
        # attributed memory message before the cache is translated to the next Agent.
        memory_prefix_len = int(generation.tokens_before)
        agent.truncate_kv_cache(memory_prefix_len)
        memory_text = self._render_memory_entry(
            agent=agent,
            persona=persona,
            response=generation.text,
            context=context,
            question=question,
            include_base=memory_prefix_len == 0,
        )
        agent.append_context_text(memory_text)
        return agent.cache_seq_len

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
    def _strip_leading_bos_for_continuation(tokenizer, rendered: str) -> str:
        bos_token = getattr(tokenizer, "bos_token", None)
        if isinstance(bos_token, str) and bos_token and rendered.startswith(bos_token):
            return rendered[len(bos_token) :]
        return rendered

    @staticmethod
    def _strip_irrelevant_chat_template_metadata(rendered: str) -> str:
        """Remove model-template metadata that is irrelevant to timeless benchmark tasks."""
        return re.sub(
            r"(?m)^[ \t]*Cutting Knowledge Date:[^\r\n]*\r?\n"
            r"[ \t]*Today Date:[^\r\n]*(?:\r?\n){1,2}",
            "",
            rendered,
        )

    @staticmethod
    def _render_chat_prompt(
        agent: Agent,
        user_content: str,
        *,
        continuation: bool,
        include_system: bool = True,
    ) -> Optional[str]:
        tokenizer = agent.model.tokenizer
        chat_template = AgentRunner._explicit_chat_template(agent)
        if chat_template is None:
            return None

        if continuation or not include_system:
            # A Memory continuation already has the original system prompt in KV.
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
                    rendered = AgentRunner._strip_irrelevant_chat_template_metadata(str(rendered))
                    if continuation:
                        rendered = AgentRunner._strip_leading_bos_for_continuation(tokenizer, rendered)
                    return rendered
            except Exception as error:
                logging.debug("Chat-template rendering failed; trying fallback/plain prompt: %s", error)
        return None

    @staticmethod
    def _render_mallm_agent_prompt(
        agent: Agent,
        system_content: str,
        user_contents: Sequence[str] | str,
        *,
        continuation: bool,
    ) -> Optional[str]:
        """Render the official Simple prompt topology for one discussion call.

        SimpleResponseGenerator produces one SYSTEM template followed by one
        USER improve/propose instruction. The reusable Memory KV prefix necessarily
        precedes these dynamic messages.
        """
        tokenizer = agent.model.tokenizer
        chat_template = AgentRunner._explicit_chat_template(agent)
        if chat_template is None:
            return None

        if isinstance(user_contents, str):
            normalized_user_contents = [user_contents]
        else:
            normalized_user_contents = list(user_contents)
        messages = [
            {"role": "system", "content": system_content.strip()},
            *(
                {"role": "user", "content": content.strip()}
                for content in normalized_user_contents
                if content.strip()
            ),
        ]
        folded_content = "\n\n".join(str(message["content"]) for message in messages)
        message_sets = [
            messages,
            [{"role": "user", "content": folded_content}],
        ]
        for candidate_messages in message_sets:
            try:
                rendered = tokenizer.apply_chat_template(
                    candidate_messages,
                    chat_template=chat_template,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if rendered:
                    rendered = AgentRunner._strip_irrelevant_chat_template_metadata(str(rendered))
                    if continuation:
                        rendered = AgentRunner._strip_leading_bos_for_continuation(tokenizer, rendered)
                    return rendered
            except Exception as error:
                logging.debug("MALLM agent prompt chat-template rendering failed; trying fallback/plain prompt: %s", error)
        return None

    def _format_agent_prompt(
        self,
        agent: Agent,
        system_content: str,
        user_contents: Sequence[str] | str,
        fallback_prompt: str,
        *,
        continuation: bool,
    ) -> str:
        rendered = self._render_mallm_agent_prompt(
            agent,
            system_content,
            user_contents,
            continuation=continuation,
        )
        return rendered if rendered else fallback_prompt

    @staticmethod
    def build_initial_prompt(
        context: str,
        question: str,
        *,
        hub_agent_id: str = "A",
        agent_count: int = 3,
    ) -> str:
        del hub_agent_id, agent_count
        return AgentRunner._build_plain_initial_prompt(context, question)

    def build_initial_prompt_for_agent(
        self,
        agent: Agent,
        context: str,
        question: str,
        *,
        hub_agent_id: str = "A",
        agent_count: int = 3,
    ) -> str:
        del hub_agent_id, agent_count
        persona = self.agent_personas[agent.node_id]
        system_content = self._build_initial_user_content(
            context,
            question,
            persona=persona,
        )
        return self._format_agent_prompt(
            agent,
            system_content,
            MALLM_SIMPLE_PROPOSE_PROMPT,
            self._build_plain_initial_prompt(
                context,
                question,
                persona=persona,
            ),
            continuation=False,
        )

    def build_followup_prompt(
        self,
        agent_id: str,
        question: str,
        *,
        context: str = "",
        current_solution: str = "",
    ) -> str:
        persona = self.agent_personas.get(agent_id, self._default_persona())
        system_content = self._general_debate_prompt(
            context,
            question,
            persona=persona,
            current_solution=current_solution,
        )
        agent = self.agents.get(agent_id)
        if agent is not None:
            return self._format_agent_prompt(
                agent,
                system_content,
                MALLM_SIMPLE_RESPONSE_PROMPT,
                self._build_plain_followup_prompt(
                    question,
                    context=context,
                    persona=persona,
                    current_solution=current_solution,
                ),
                continuation=True,
            )
        return self._build_plain_followup_prompt(
            question,
            context=context,
            persona=persona,
            current_solution=current_solution,
        )

    def build_initial_verification_prompt(
        self,
        agent_id: str,
        question: str,
        *,
        context: str,
        previous_response: str,
        reason: str,
    ) -> str:
        """Rewrite an invalid first proposal without introducing AGREE/DISAGREE."""
        persona = self.agent_personas.get(agent_id, self._default_persona())
        system_content = self._general_debate_prompt(
            context,
            question,
            persona=persona,
        )
        previous_preview = previous_response.strip()
        if len(previous_preview) > 1600:
            previous_preview = previous_preview[-1600:]
        user_content = (
            "The previous proposal did not provide a valid, unambiguous StrategyQA answer. Rewrite it from scratch.\n"
            f"Failure: {reason}\n"
            "Rules:\n"
            "1. Give concise reasoning that supports exactly one Yes/No answer.\n"
            "2. Do not use [AGREE] or [DISAGREE] on the first proposal.\n"
            "3. End with exactly one of: `Final Solution: Yes` or `Final Solution: No`.\n"
            "4. Do not state the opposite answer as the final conclusion.\n\n"
            "Previous invalid proposal:\n"
            f"{previous_preview}"
        )
        fallback = f"{system_content}\n\n{user_content}\n### Response:\n"
        agent = self.agents.get(agent_id)
        if agent is None:
            return fallback
        return self._format_agent_prompt(
            agent,
            system_content,
            user_content,
            fallback,
            continuation=False,
        )

    def build_verification_prompt(
        self,
        agent_id: str,
        question: str,
        *,
        context: str,
        current_solution: str,
        current_answer: Optional[str],
        previous_response: str,
        reason: str,
    ) -> str:
        """Ask the same Agent to rewrite an internally inconsistent response.

        This verifier is only entered after the official Simple response failed a
        deterministic StrategyQA consistency check. It does not change the four
        MALLM components; it validates their generated agent message before that
        agent message is committed to shared Memory / Supermajority Consensus.
        """
        persona = self.agent_personas.get(agent_id, self._default_persona())
        system_content = self._general_debate_prompt(
            context,
            question,
            persona=persona,
            current_solution=current_solution,
        )
        previous_preview = previous_response.strip()
        if len(previous_preview) > 1600:
            previous_preview = previous_preview[-1600:]
        user_content = (
            "Verification failed because the previous response is internally inconsistent or incomplete. Rewrite it from scratch.\n"
            f"Failure: {reason}\n"
            f"Current answer: {current_answer or 'UNKNOWN'}\n"
            "Rules:\n"
            "1. Start the first line with exactly [AGREE] or [DISAGREE].\n"
            f"2. [AGREE] means you endorse the current solution; if you state a final answer, it must stay exactly {current_answer or 'the current answer'}.\n"
            "3. [DISAGREE] means you reject or materially improve the current solution. Your final Yes/No may stay the same if you are correcting the reasoning, or change if the conclusion is wrong.\n"
            "4. The reasoning must support the marker and the final Yes/No answer.\n"
            "5. End with exactly one of: `Final Solution: Yes` or `Final Solution: No`.\n"
            "6. Keep the rewrite concise; do not discuss these verification rules.\n\n"
            "Previous invalid response:\n"
            f"{previous_preview}"
        )
        fallback = f"{system_content}\n\n{user_content}\n### Response:\n"
        agent = self.agents.get(agent_id)
        if agent is None:
            return fallback
        return self._format_agent_prompt(
            agent,
            system_content,
            user_content,
            fallback,
            continuation=True,
        )

    @staticmethod
    def _append_agent_message_to_transcript(transcript: str, agent_id: str, response: str) -> str:
        clean_response = response.strip() or "[empty]"
        return f"{transcript}{clean_response}\n"

    @staticmethod
    def extract_final_answer(transcript: str, selected_solution: str) -> str:
        del transcript
        answer = AgentRunner._extract_strategyqa_answer(selected_solution)
        if answer is not None:
            return answer
        return postprocess_generated_answer((selected_solution or "").strip())

    @staticmethod
    def _preview_text(text: str, max_chars: int) -> str:
        clean = re.sub(r"\s+", " ", (text or "").strip())
        if len(clean) <= max_chars:
            return clean
        return clean[: max(0, max_chars - 3)] + "..."

    def _agent_at_sequence(self, sequence_index: int) -> Agent:
        return self.agent_sequence[sequence_index % len(self.agent_sequence)]

    @staticmethod
    def _tensor_storage_key(tensor: torch.Tensor) -> Tuple[str, Optional[int], int]:
        storage = tensor.untyped_storage()
        return (tensor.device.type, tensor.device.index, int(storage.data_ptr()))

    @classmethod
    def _past_storage_bytes(
        cls,
        past_key_values: Optional[PastKeyValues],
        seen: set,
        *,
        device_index: Optional[int],
    ) -> int:
        if past_key_values is None:
            return 0
        total = 0
        for key, value in past_key_values:
            for tensor in (key, value):
                if not isinstance(tensor, torch.Tensor):
                    continue
                if device_index is not None:
                    if tensor.device.type != "cuda":
                        continue
                    tensor_device_index = torch.cuda.current_device() if tensor.device.index is None else tensor.device.index
                    if tensor_device_index != device_index:
                        continue
                storage = tensor.untyped_storage()
                storage_key = cls._tensor_storage_key(tensor)
                if storage_key in seen:
                    continue
                seen.add(storage_key)
                total += int(storage.nbytes())
        return total

    def _live_kv_cache_objects(self) -> List[PastKeyValues]:
        objects: List[PastKeyValues] = []
        for agent in self.agent_sequence:
            if agent.past_key_values is not None:
                objects.append(agent.past_key_values)
            objects.extend(agent.pretranslated_past_by_edge.values())
        objects.extend(
            past for _edge_id, past, _token_ids in self._pending_pretranslated_second_hops.values()
        )
        return objects

    def _live_kv_cache_bytes(self) -> int:
        """Return CUDA bytes owned by all live AgentRunner KV/cache tensors.

        Resident Agent KV, pretranslated edge KV, and pending second-hop KV are
        counted exactly once by underlying storage on this runner's CUDA device.
        On CUDA runs, CPU-offloaded cache tensors are intentionally excluded.
        On CPU-only runs/tests, all cache storage is counted for diagnostics.
        """
        device_index: Optional[int] = None
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            device_obj = torch.device(self.device)
            device_index = torch.cuda.current_device() if device_obj.index is None else device_obj.index
        seen: set = set()
        total = 0
        for past in self._live_kv_cache_objects():
            total += self._past_storage_bytes(past, seen, device_index=device_index)
        return int(total)

    def _update_peak_memory_breakdown(self) -> None:
        live_kv_cache_bytes = self._live_kv_cache_bytes()
        self._kv_cache_memory_samples_bytes.append(live_kv_cache_bytes)
        self._peak_kv_cache_bytes = max(self._peak_kv_cache_bytes, live_kv_cache_bytes)

        measured = measure_gpu_memory_breakdown_bytes(
            self.device,
            models=self.ctx.tp.models.values(),
            translator_pool=self.translator_pool,
            kv_objects=self._live_kv_cache_objects(),
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
        if not self.log_agents:
            return
        label = "?" if example_index is None else str(example_index)
        agent_order = "->".join(agent.node_id for agent in self.agent_sequence)
        persona_summary = " | ".join(
            f"{agent_id}={self._role_text(persona)}" for agent_id, persona in self.agent_personas.items()
        )
        logging.info(
            "\n[AgentRunner] ===== Example %s =====\n"
            "  config   | mode=%s | alg=%s | agents=%s | discussion=memory | response=simple | decision=supermajority_then_majority_vote\n"
            "  order    | %s cyclic | max_turns=%s\n"
            "  personas | %s\n"
            "  question | %s\n"
            "  gold     | %s",
            label,
            self.cache_mode,
            self.alg,
            len(self.agent_sequence),
            agent_order,
            self.max_turns,
            persona_summary,
            self._preview_text(question, self.log_max_chars),
            list(gold_answers)[:3],
        )

    def _log_agent(self, *, example_index: Optional[int], sequence_index: int, record: AgentMessageRecord) -> None:
        if not self.log_agents:
            return
        label = "?" if example_index is None else str(example_index)
        turn_number = sequence_index + 1
        logging.info(
            "[AgentRunner][Example %s][Turn %d/%d][Agent %s]\n"
            "  route    | mode=%s | agent=%s | hub=%s | translated=%s(%s) | offload=%s(%s)\n"
            "  decision | state=%s | solution=%s\n"
            "  verify   | passed=%s | retries=%d | syntax=%s(%s; failures=%d) | semantic=%s(%s; failures=%d) | marker=%s | final_answer=%s | reason=%s\n"
            "  prompt   | %s\n"
            "  response | %s",
            label,
            turn_number,
            self.max_turns,
            record.agent_id,
            record.cache_mode,
            record.agent_id,
            record.is_hub,
            record.translated_edge_id or "null",
            record.translated_offload_kind or "none",
            record.offload_edge_id or "null",
            record.offload_kind or "none",
            record.response_state or "pending",
            self._preview_text(record.solution or "", self.log_max_chars),
            record.verification_passed,
            int(record.verification_attempts),
            record.verification_syntax_passed,
            self._preview_text(record.verification_syntax_reason or "none", self.log_max_chars),
            int(record.verification_syntax_failures),
            record.verification_semantic_passed,
            self._preview_text(record.verification_semantic_reason or "none", self.log_max_chars),
            int(record.verification_semantic_failures),
            record.agreement_marker or "none",
            record.final_answer or "none",
            self._preview_text(record.verification_reason or "none", self.log_max_chars),
            self._preview_text(record.prompt, self.log_max_chars),
            self._preview_text(record.response, self.log_max_chars),
        )

    def _log_example_end(self, *, example_index: Optional[int], prediction: str, accuracy: float) -> None:
        if not self.log_agents:
            return
        label = "?" if example_index is None else str(example_index)
        logging.info(
            "[AgentRunner][Example %s] result | mode=%s | prediction=%s | accuracy=%.4f",
            label,
            self.cache_mode,
            self._preview_text(prediction, self.log_max_chars),
            accuracy,
        )

    def _synchronize_ttft_device(self) -> None:
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            device_obj = torch.device(self.device)
            device_index = torch.cuda.current_device() if device_obj.index is None else device_obj.index
            torch.cuda.synchronize(device_index)

    def _measure_ttft_phase(self, fn):
        """Measure one required pre-first-token phase without changing its inputs.

        Used only for cache translation/offload phases. Synchronization makes the
        measured wall time belong to this phase rather than adjacent CUDA work.
        Verification, logging, and memory-accounting work are intentionally not
        measured as part of TTFT.
        """
        self._synchronize_ttft_device()
        started_at = time.perf_counter()
        result = fn()
        self._synchronize_ttft_device()
        return result, time.perf_counter() - started_at

    def _generate_discussion_attempt(
        self,
        *,
        agent: Agent,
        prompt: str,
        measure_ttft: bool,
    ) -> Tuple[AgentGeneration, Optional[float]]:
        """Generate one discussion attempt and optionally capture first-token time."""
        previous_flag = bool(getattr(agent, "measure_first_token_ttft", False))
        agent.measure_first_token_ttft = bool(measure_ttft)
        try:
            generation = agent.generate_response(prompt)
        finally:
            agent.measure_first_token_ttft = previous_flag
        return generation, generation.first_token_ttft_sec if measure_ttft else None

    def _should_clear_source_after_offload(self, source_agent: Agent) -> bool:
        # In free mode, the non-hub agent that just transmitted its newly generated
        # cache delta is freed. The hub stays resident so it can always receive and
        # accumulate deltas from the other agent(s).
        return self.cache_mode == CACHE_MODE_FREE and source_agent.node_id != self.hub_agent.node_id

    def _prepare_outgoing_route_translation(self, *, source_agent: Agent, logical_target_agent: Agent) -> None:
        """Pretranslate the physical star-topology route before the next handoff.

        A zero-length first hop is valid when source and target already own the
        same shared-memory prefix (for example after a rejected Verification
        attempt is rolled back). In that case no translation is prepared for that
        hop; for non-hub -> non-hub routing the unchanged hub cache is used to
        prepare the second hop.
        """
        self._pending_pretranslated_second_hops.pop((source_agent.node_id, logical_target_agent.node_id), None)
        if source_agent.past_key_values is None:
            return

        source_is_hub = source_agent.node_id == self.hub_agent.node_id
        target_is_hub = logical_target_agent.node_id == self.hub_agent.node_id
        if source_is_hub or target_is_hub:
            prefix_tokens, expected_delta_tokens, _ = self._build_missing_cache_delta(
                source_agent=source_agent,
                target_agent=logical_target_agent,
            )
            if expected_delta_tokens == 0:
                return
            self.cache_translator.refresh_pretranslated_cache(
                source_agent=source_agent,
                target_agent=logical_target_agent,
                prefix_tokens=(prefix_tokens if self.cache_mode == CACHE_MODE_RETAIN else 0),
            )
            return

        # First physical hop: source non-hub -> hub.  If it is a no-op, the
        # future hub cache is simply its current resident cache.
        prefix_tokens, expected_delta_tokens, _ = self._build_missing_cache_delta(
            source_agent=source_agent,
            target_agent=self.hub_agent,
        )
        if expected_delta_tokens == 0:
            if self.hub_agent.past_key_values is None:
                raise RuntimeError(
                    f"Zero-delta route {source_agent.node_id}->{self.hub_agent.node_id} "
                    "requires a resident hub cache."
                )
            future_hub_past = self.hub_agent.past_key_values
        else:
            self.cache_translator.refresh_pretranslated_cache(
                source_agent=source_agent,
                target_agent=self.hub_agent,
                prefix_tokens=(prefix_tokens if self.cache_mode == CACHE_MODE_RETAIN else 0),
            )
            first_delta_piece, _ = self.cache_translator._slice_pretranslated_cache_piece(
                source_agent=source_agent,
                target_agent=self.hub_agent,
                prefix_tokens=prefix_tokens,
                expected_delta_tokens=expected_delta_tokens,
            )
            if get_past_seq_len(first_delta_piece) != expected_delta_tokens:
                raise ValueError(
                    f"Prepared source->hub delta length mismatch on {source_agent.node_id}->{self.hub_agent.node_id}: "
                    f"expected_delta_tokens={expected_delta_tokens} "
                    f"piece_tokens={get_past_seq_len(first_delta_piece)}"
                )
            if self.hub_agent.past_key_values is None:
                future_hub_past = first_delta_piece
            else:
                future_hub_past = _concat_past_key_values(self.hub_agent.past_key_values, first_delta_piece)

        # Second physical hop: future hub -> logical target.  Free keeps the
        # original full-cache pretranslation path unchanged.  Retain translates
        # only the logical target's missing suffix.
        second_source_past = future_hub_past
        second_source_token_ids = list(source_agent.cache_token_ids)
        if self.cache_mode == CACHE_MODE_RETAIN:
            second_prefix_tokens, second_delta_tokens, _ = self._build_missing_cache_delta(
                source_agent=source_agent,
                target_agent=logical_target_agent,
            )
            if second_delta_tokens == 0:
                return
            if second_prefix_tokens:
                second_source_past = slice_past_suffix(future_hub_past, second_prefix_tokens)
                second_source_token_ids = second_source_token_ids[second_prefix_tokens:]

        second_edge_id, second_target_past = self.cache_translator.build_pretranslated_past_for_edge(
            source_agent=self.hub_agent,
            target_agent=logical_target_agent,
            source_past_key_values=second_source_past,
            source_token_ids=second_source_token_ids,
        )
        self._pending_pretranslated_second_hops[(source_agent.node_id, logical_target_agent.node_id)] = (
            second_edge_id,
            second_target_past,
            second_source_token_ids,
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
        if expected_delta_tokens < 0:
            raise ValueError(
                f"Invalid negative KV delta on {source_agent.node_id}->{target_agent.node_id}: "
                f"source_tokens={len(source_ids)} target_prefix_tokens={prefix_tokens}"
            )
        # A zero-length delta is valid. It occurs naturally when an agent message is
        # rejected by Verification and the source Agent is rolled back to the same
        # shared-memory prefix that the next target already owns.
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
        if expected_delta_tokens == 0:
            # No replay/translation is necessary: target already owns the exact
            # logical prefix represented by source.  In free mode the non-hub
            # source can still be released because ownership has already moved.
            edge = self.cache_translator._get_edge(source_agent.node_id, target_agent.node_id)
            metadata = {
                "mode": "offload",
                "offload_kind": OFFLOAD_KIND_NOOP,
                "edge_id": edge.id,
                "tokens_sent": 0,
                "tokens_received": 0,
                "target_piece_tokens": 0,
                "target_tokens_before_replay": int(target_tokens_before_replay),
                "target_tokens_after_replay": int(target_agent.cache_seq_len),
                "expected_delta_tokens": 0,
                "delta_prefix_matched": bool(delta_prefix_matched),
            }
            cleared = False
            if self._should_clear_source_after_offload(source_agent):
                if source_agent.past_key_values is not None:
                    source_agent.clear_kv_cache(empty_cuda_cache=True)
                    cleared = True
            return metadata, cleared
        return self._offload_into(
            source_agent=source_agent,
            target_agent=target_agent,
            target_tokens_before_replay=target_tokens_before_replay,
            expected_delta_tokens=expected_delta_tokens,
            delta_prefix_matched=delta_prefix_matched,
        )

    def _star_offload_to_agent(
        self,
        *,
        source_agent: Agent,
        target_agent: Agent,
    ) -> Tuple[Agent, Dict[str, Any], bool, Agent, Dict[str, Any]]:
        """Route KV handoff through the hub when both endpoints are non-hub.

        Returns:
            record_target_agent: the physical target of source_agent's outgoing
                offload, used for source_agent's agent message record.
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

        first_meta, source_cleared = self._offload_delta_hop(
            source_agent=source_agent,
            target_agent=self.hub_agent,
        )

        second_prefix_tokens, second_delta_tokens, _ = self._build_missing_cache_delta(
            source_agent=self.hub_agent,
            target_agent=target_agent,
        )
        if self.cache_mode == CACHE_MODE_FREE:
            # Preserve the original free-mode behavior exactly: the pending second
            # hop is a full translated hub cache regardless of the target prefix.
            if pending_second_hop is None:
                raise RuntimeError(
                    f"Missing pretranslated second-hop cache for {source_agent.node_id}->"
                    f"{self.hub_agent.node_id}->{target_agent.node_id}."
                )
            second_edge_id, second_target_past, second_token_ids = pending_second_hop
            expected_prepared_ids = list(self.hub_agent.cache_token_ids)
            if list(second_token_ids) != expected_prepared_ids:
                raise RuntimeError(
                    f"Prepared second-hop cache is stale for {second_edge_id}: "
                    f"prepared_tokens={len(second_token_ids)} expected_tokens={len(expected_prepared_ids)}"
                )
            self.hub_agent.set_pretranslated_cache(
                edge_id=second_edge_id,
                past_key_values=second_target_past,
                cache_token_ids=second_token_ids,
            )
        elif second_delta_tokens > 0:
            if pending_second_hop is None:
                raise RuntimeError(
                    f"Missing pretranslated second-hop cache for {source_agent.node_id}->"
                    f"{self.hub_agent.node_id}->{target_agent.node_id}."
                )
            second_edge_id, second_target_past, second_token_ids = pending_second_hop
            expected_prepared_ids = list(self.hub_agent.cache_token_ids)[second_prefix_tokens:]
            if list(second_token_ids) != expected_prepared_ids:
                raise RuntimeError(
                    f"Prepared second-hop cache is stale for {second_edge_id}: "
                    f"prepared_tokens={len(second_token_ids)} expected_tokens={len(expected_prepared_ids)}"
                )
            self.hub_agent.set_pretranslated_cache(
                edge_id=second_edge_id,
                past_key_values=second_target_past,
                cache_token_ids=second_token_ids,
            )
        second_meta, _ = self._offload_delta_hop(
            source_agent=self.hub_agent,
            target_agent=target_agent,
        )
        return self.hub_agent, first_meta, source_cleared, self.hub_agent, second_meta

    def _tokens_sent_check_passed(self, record: AgentMessageRecord, offload_meta: Dict[str, Any]) -> bool:
        # Every handoff is a delta. The expected amount is
        # source_cache_tokens_after_generation - target_cache_tokens_before_replay.
        tokens_sent = int(offload_meta.get("tokens_sent", 0))
        expected_delta_tokens = int(offload_meta.get("expected_delta_tokens", 0))
        return tokens_sent == expected_delta_tokens

    def _apply_offload_metadata_to_record(
        self,
        record: AgentMessageRecord,
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

    def _generate_verified_initial(
        self,
        *,
        agent: Agent,
        initial_prompt: str,
        context: str,
        question: str,
    ) -> Tuple[AgentGeneration, StrategyQAVerificationResult, int, VerificationStageCounts, Optional[float]]:
        """Generate the first proposal until strict syntax and semantics both pass.

        Invalid attempts are rolled back to the exact pre-prompt KV prefix and
        retried without a retry limit. No invalid response is ever accepted as a
        fallback merely because a retry budget was exhausted.
        """
        prompt = initial_prompt
        syntax_failures = 0
        semantic_failures = 0
        previous_response = ""
        reason = "not checked"
        attempt = 0
        turn_generation_ttft_sec: Optional[float] = None

        while True:
            if attempt > 0:
                prompt = self.build_initial_verification_prompt(
                    agent.node_id,
                    question,
                    context=context,
                    previous_response=previous_response,
                    reason=reason,
                )
            self._set_generation_seed(
                "discussion",
                turn_index=1,
                agent_index=0,
                attempt=attempt,
            )
            generation, measured_ttft = self._generate_discussion_attempt(
                agent=agent,
                prompt=prompt,
                measure_ttft=attempt == 0,
            )
            if attempt == 0:
                turn_generation_ttft_sec = measured_ttft
            self._update_peak_memory_breakdown()

            # Stage 1: deterministic syntax. Do not spend a semantic-verifier
            # generation on malformed output.
            syntax = self._verify_initial_syntax(generation.text)
            if syntax.syntax_passed and syntax.final_answer is not None:
                self._set_generation_seed(
                    "verification",
                    turn_index=1,
                    agent_index=0,
                    attempt=attempt,
                )
                _semantic_raw, semantic_passed = self._verify_response_semantics_with_model(
                    agent=agent,
                    context=context,
                    question=question,
                    response=generation.text,
                    marker=None,
                    final_answer=syntax.final_answer,
                )
                semantic_reason = (
                    "ok"
                    if semantic_passed is True
                    else "semantic verifier rejected internally inconsistent initial reasoning"
                )
                verification = StrategyQAVerificationResult(
                    passed=semantic_passed is True,
                    syntax_passed=True,
                    syntax_reason="ok",
                    semantic_passed=semantic_passed is True,
                    semantic_reason=semantic_reason,
                    marker=None,
                    final_answer=syntax.final_answer,
                )
                if verification.passed:
                    return (
                        generation,
                        verification,
                        attempt,
                        VerificationStageCounts(syntax_failures, semantic_failures),
                        turn_generation_ttft_sec,
                    )
                semantic_failures += 1
                reason = verification.reason
            else:
                syntax_failures += 1
                verification = syntax
                reason = syntax.reason

            agent.truncate_kv_cache(generation.tokens_before)
            self._update_peak_memory_breakdown()
            previous_response = generation.text
            attempt += 1

    def _generate_verified_followup(
        self,
        *,
        agent: Agent,
        initial_prompt: str,
        context: str,
        question: str,
        current_solution: str,
        current_answer: Optional[str],
        current_response: Optional[str],
        turn_index: int,
        agent_index: int,
    ) -> Tuple[AgentGeneration, StrategyQAVerificationResult, int, VerificationStageCounts, Optional[float]]:
        """Generate a follow-up until strict syntax and semantics both pass.

        Invalid attempts are rolled back to the exact pre-prompt KV prefix, so they
        never enter shared Memory or Supermajority Consensus. Corrective retries are
        unbounded and deterministically seeded by the retry index.
        """
        prompt = initial_prompt
        syntax_failures = 0
        semantic_failures = 0
        previous_response = ""
        reason = "not checked"
        attempt = 0
        turn_generation_ttft_sec: Optional[float] = None

        while True:
            if attempt > 0:
                prompt = self.build_verification_prompt(
                    agent.node_id,
                    question,
                    context=context,
                    current_solution=current_solution,
                    current_answer=current_answer,
                    previous_response=previous_response,
                    reason=reason,
                )

            self._set_generation_seed(
                "discussion",
                turn_index=turn_index,
                agent_index=agent_index,
                attempt=attempt,
            )
            generation, measured_ttft = self._generate_discussion_attempt(
                agent=agent,
                prompt=prompt,
                measure_ttft=attempt == 0,
            )
            if attempt == 0:
                turn_generation_ttft_sec = measured_ttft
            self._update_peak_memory_breakdown()

            # Stage 1: deterministic syntax only. A malformed marker/final line
            # is immediately retried and never reaches the semantic model.
            syntax = self._verify_followup_syntax(generation.text)
            if syntax.syntax_passed:
                claimed_final_answer = syntax.final_answer
                if claimed_final_answer is None and syntax.marker == "agree":
                    claimed_final_answer = current_answer
                if claimed_final_answer not in {"Yes", "No"}:
                    verification = StrategyQAVerificationResult(
                        passed=False,
                        syntax_passed=True,
                        syntax_reason="ok",
                        semantic_passed=False,
                        semantic_reason="semantic verification has no concrete claimed Yes/No answer",
                        marker=syntax.marker,
                        final_answer=None,
                    )
                    reason = verification.reason
                    agent.truncate_kv_cache(generation.tokens_before)
                    self._update_peak_memory_breakdown()
                    previous_response = generation.text
                    semantic_failures += 1
                    attempt += 1
                    continue

                # Deterministic semantic contract checks belong to stage 2. They
                # run before the model-based internal-consistency verifier.
                if syntax.marker == "agree" and claimed_final_answer != current_answer:
                    semantic_passed = False
                    semantic_reason = (
                        f"[AGREE] final answer {claimed_final_answer} does not match current answer {current_answer}"
                    )
                else:
                    self._set_generation_seed(
                        "verification",
                        turn_index=turn_index,
                        agent_index=agent_index,
                        attempt=attempt,
                    )
                    _semantic_raw, semantic_passed = self._verify_response_semantics_with_model(
                        agent=agent,
                        context=context,
                        question=question,
                        response=generation.text,
                        marker=syntax.marker,
                        final_answer=claimed_final_answer,
                        current_answer=current_answer,
                        current_response=current_response,
                    )
                    semantic_reason = (
                        "ok"
                        if semantic_passed is True
                        else "semantic verifier rejected marker/reasoning/final-answer consistency"
                    )
                verification = StrategyQAVerificationResult(
                    passed=semantic_passed is True,
                    syntax_passed=True,
                    syntax_reason="ok",
                    semantic_passed=semantic_passed is True,
                    semantic_reason=semantic_reason,
                    marker=syntax.marker,
                    final_answer=claimed_final_answer,
                )
                if verification.passed:
                    return (
                        generation,
                        verification,
                        attempt,
                        VerificationStageCounts(syntax_failures, semantic_failures),
                        turn_generation_ttft_sec,
                    )
                semantic_failures += 1
                reason = verification.reason
            else:
                syntax_failures += 1
                verification = syntax
                reason = syntax.reason

            # Do not let an invalid control prompt/response contaminate Memory.
            # Restore exactly the KV prefix that existed before this attempt.
            agent.truncate_kv_cache(generation.tokens_before)
            self._update_peak_memory_breakdown()
            previous_response = generation.text
            attempt += 1

    def _run_offload_turns(
        self,
        *,
        transcript: str,
        initial_generation: AgentGeneration,
        agent_messages: List[AgentMessageRecord],
        turns: List[AgentTurnRecord],
        context: str,
        question: str,
        example_index: Optional[int],
        initial_pretranslation_sec: float = 0.0,
    ) -> Tuple[str, str]:
        # The verified first proposal has already passed syntax and semantic
        # consistency checks. Use the answer captured by verification rather than
        # reparsing arbitrary reasoning text here.
        current_answer = agent_messages[0].final_answer
        if current_answer not in {"Yes", "No"}:
            raise RuntimeError("verified initial proposal lost its Final Solution answer")
        current_solution = current_answer
        current_response = initial_generation.text
        agent_messages[0].final_answer = current_answer
        agent_messages[0].solution = current_solution
        agent_messages[0].response_state = "draft"

        # Each Agent contributes at most one latest verified semantic vote. The
        # initial proposal is Turn 1. The first pass through all configured Agents is
        # mandatory: consensus is not evaluated until every Agent has completed one
        # verified Turn. From that point onward, it is evaluated after every Turn.
        agent_votes: Dict[str, str] = {self.hub_agent.node_id: current_answer}
        self._consensus_reached = False
        self._consensus_turn = None
        self._consensus_answer = None
        self._final_agent_votes = dict(agent_votes)
        self._evaluate_turn_consensus(
            turn_index=1,
            agent_id=self.hub_agent.node_id,
            agent_votes=agent_votes,
            turns=turns,
        )

        agent_count = len(self.agent_sequence)
        sequence_index = 1
        pending_pretranslation_sec = float(initial_pretranslation_sec)
        while sequence_index < self.max_turns and not self._consensus_reached:
            current_source = self._agent_at_sequence(sequence_index - 1)
            current_target = self._agent_at_sequence(sequence_index)
            source_record = agent_messages[-1]

            edge_id: Optional[str] = None
            incoming_offload_kind: Optional[str] = None
            tokens_received = 0
            offload_sec = 0.0
            if current_source.node_id != current_target.node_id:
                (
                    (
                        record_offload_target,
                        source_offload_meta,
                        source_cache_cleared,
                        incoming_from_agent,
                        incoming_meta,
                    ),
                    offload_sec,
                ) = self._measure_ttft_phase(
                    lambda: self._star_offload_to_agent(
                        source_agent=current_source,
                        target_agent=current_target,
                    )
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
            self._log_agent(
                example_index=example_index,
                sequence_index=sequence_index - 1,
                record=source_record,
            )

            prompt = self.build_followup_prompt(
                current_target.node_id,
                question,
                context=context,
                current_solution=current_solution,
            )
            turn_index = sequence_index + 1
            agent_index = sequence_index % agent_count
            (
                generation,
                verification,
                verification_attempts,
                verification_stage_counts,
                generation_ttft_sec,
            ) = self._generate_verified_followup(
                agent=current_target,
                initial_prompt=prompt,
                context=context,
                question=question,
                current_solution=current_solution,
                current_answer=current_answer,
                current_response=current_response,
                turn_index=turn_index,
                agent_index=agent_index,
            )

            record = self._agent_message_record(
                generation,
                translated_edge_id=edge_id,
                translated_offload_kind=incoming_offload_kind,
                tokens_received=tokens_received,
            )
            record.ttft_sec = (
                None
                if generation_ttft_sec is None
                else float(pending_pretranslation_sec + offload_sec + generation_ttft_sec)
            )
            pending_pretranslation_sec = 0.0
            record.verification_attempts = verification_attempts
            record.verification_syntax_failures = verification_stage_counts.syntax_failures
            record.verification_semantic_failures = verification_stage_counts.semantic_failures
            record.verification_passed = verification.passed
            record.verification_reason = verification.reason
            record.verification_syntax_passed = verification.syntax_passed
            record.verification_syntax_reason = verification.syntax_reason
            record.verification_semantic_passed = verification.semantic_passed
            record.verification_semantic_reason = verification.semantic_reason
            record.agreement_marker = verification.marker
            record.final_answer = verification.final_answer

            if not verification.passed:
                raise RuntimeError(
                    f"Internal error: unbounded verification returned an invalid response for agent {current_target.node_id}"
                )

            memory_tokens_after = self._commit_discussion_memory(
                agent=current_target,
                generation=generation,
                persona=self.agent_personas[current_target.node_id],
                context=context,
                question=question,
            )
            self._update_peak_memory_breakdown()
            transcript = self._append_agent_message_to_transcript(
                transcript + generation.prompt_text,
                current_target.node_id,
                generation.text,
            )
            previous_solution = current_solution
            previous_answer = current_answer
            if verification.marker == "agree":
                current_solution = previous_solution
                current_answer = previous_answer
                response_state = "agree"
            else:
                current_solution = str(verification.final_answer)
                current_answer = verification.final_answer
                response_state = "revise"
            current_response = generation.text
            record.memory_tokens_after = memory_tokens_after
            record.solution = current_solution
            record.response_state = response_state

            if current_answer in {"Yes", "No"}:
                agent_votes[current_target.node_id] = current_answer

            agent_messages.append(record)
            self._evaluate_turn_consensus(
                turn_index=turn_index,
                agent_id=current_target.node_id,
                agent_votes=agent_votes,
                turns=turns,
            )
            sequence_index += 1

            if not self._consensus_reached and sequence_index < self.max_turns:
                next_target = self._agent_at_sequence(sequence_index)
                if current_target.node_id != next_target.node_id:
                    _, pending_pretranslation_sec = self._measure_ttft_phase(
                        lambda: self._prepare_outgoing_route_translation(
                            source_agent=current_target,
                            logical_target_agent=next_target,
                        )
                    )
                    # Pretranslation materializes one or more full target-side KV
                    # caches. Measure immediately; the next handoff may consume or
                    # free them before the old sampling point and would undercount
                    # the true live KV peak.
                    self._update_peak_memory_breakdown()

        self._update_peak_memory_breakdown()
        self._log_agent(
            example_index=example_index,
            sequence_index=len(agent_messages) - 1,
            record=agent_messages[-1],
        )

        # After every Agent has participated once, a >66% supermajority wins
        # immediately after any subsequent completed Turn (including the Turn that
        # completes first participation). If max_turns expires without one, select
        # from the Agents' latest verified votes by
        # majority; only an exact tie is resolved randomly. There is no response
        # fallback path.
        if self._consensus_reached:
            final_solution = self._consensus_answer
            self._final_decision_method = "supermajority_consensus"
        else:
            final_solution, self._final_decision_method = self._majority_vote_with_random_tie(
                agent_votes
            )
        self._final_decision_answer = final_solution
        return transcript, str(final_solution).strip()

    def _run_retain_turns(
        self,
        *,
        transcript: str,
        initial_generation: AgentGeneration,
        agent_messages: List[AgentMessageRecord],
        turns: List[AgentTurnRecord],
        context: str,
        question: str,
        example_index: Optional[int],
        initial_pretranslation_sec: float = 0.0,
    ) -> Tuple[str, str]:
        return self._run_offload_turns(
            transcript=transcript,
            initial_generation=initial_generation,
            agent_messages=agent_messages,
            turns=turns,
            context=context,
            question=question,
            example_index=example_index,
            initial_pretranslation_sec=initial_pretranslation_sec,
        )

    def _run_free_turns(
        self,
        *,
        transcript: str,
        initial_generation: AgentGeneration,
        agent_messages: List[AgentMessageRecord],
        turns: List[AgentTurnRecord],
        context: str,
        question: str,
        example_index: Optional[int],
        initial_pretranslation_sec: float = 0.0,
    ) -> Tuple[str, str]:
        return self._run_offload_turns(
            transcript=transcript,
            initial_generation=initial_generation,
            agent_messages=agent_messages,
            turns=turns,
            context=context,
            question=question,
            example_index=example_index,
            initial_pretranslation_sec=initial_pretranslation_sec,
        )

    def _run_example_impl(
        self,
        *,
        context: str,
        question: str,
        gold_answers: Sequence[str],
        example_index: Optional[int] = None,
    ) -> AgentRunnerResult:
        example_seed = self.seed if example_index is None else self.seed + max(0, int(example_index) - 1)
        self._current_example_seed = int(example_seed)
        set_seed(example_seed)
        self._peak_memory_breakdown_bytes = None
        self._peak_kv_cache_bytes = 0
        self._kv_cache_memory_samples_bytes = []
        self._consensus_reached = False
        self._consensus_turn = None
        self._consensus_answer = None
        self._final_agent_votes = {}
        self._final_decision_method = None
        self._final_decision_answer = None
        for agent in self.agent_sequence:
            agent.reset()
        self._pending_pretranslated_second_hops.clear()
        # Release allocator cache from the previous example. Memory accounting
        # itself is based only on Model, Translator, and live KV tensor storage.
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        self.translator_pool.eval()
        for node in self.ctx.nodes:
            self.ctx.tp.get_model(node.id).eval()

        self.agent_personas = self._generate_expert_personas(context, question)
        self._log_example_start(question=question, gold_answers=gold_answers, example_index=example_index)
        transcript = self.build_initial_prompt_for_agent(
            self.hub_agent,
            context,
            question,
            hub_agent_id=self.hub_agent.node_id,
            agent_count=len(self.agent_sequence),
        )
        agent_messages: List[AgentMessageRecord] = []
        turns: List[AgentTurnRecord] = []

        # The first node is the HubAgent and starts from native Base Context + Prompt.
        # Its sampling stream must not depend on how many persona generations ran.
        # Verify that the proposal actually states one StrategyQA Yes/No answer before it
        # is committed to shared Memory; otherwise retry deterministically.
        (
            generation,
            initial_verification,
            initial_verification_attempts,
            initial_verification_stage_counts,
            initial_generation_ttft_sec,
        ) = self._generate_verified_initial(
            agent=self.hub_agent,
            initial_prompt=transcript,
            context=context,
            question=question,
        )
        if not initial_verification.passed:
            raise RuntimeError("Internal error: unbounded initial verification returned an invalid response")
        initial_memory_tokens = self._commit_discussion_memory(
            agent=self.hub_agent,
            generation=generation,
            persona=self.agent_personas[self.hub_agent.node_id],
            context=context,
            question=question,
        )
        initial_pretranslation_sec = 0.0
        if self.max_turns > 1:
            next_target = self._agent_at_sequence(1)
            if self.hub_agent.node_id != next_target.node_id:
                _, initial_pretranslation_sec = self._measure_ttft_phase(
                    lambda: self._prepare_outgoing_route_translation(
                        source_agent=self.hub_agent,
                        logical_target_agent=next_target,
                    )
                )
        self._update_peak_memory_breakdown()
        transcript = self._append_agent_message_to_transcript(transcript, self.hub_agent.node_id, generation.text)
        record = self._agent_message_record(generation)
        record.ttft_sec = initial_generation_ttft_sec
        record.memory_tokens_after = initial_memory_tokens
        record.verification_attempts = initial_verification_attempts
        record.verification_syntax_failures = initial_verification_stage_counts.syntax_failures
        record.verification_semantic_failures = initial_verification_stage_counts.semantic_failures
        record.verification_passed = initial_verification.passed
        record.verification_reason = initial_verification.reason
        record.verification_syntax_passed = initial_verification.syntax_passed
        record.verification_syntax_reason = initial_verification.syntax_reason
        record.verification_semantic_passed = initial_verification.semantic_passed
        record.verification_semantic_reason = initial_verification.semantic_reason
        record.agreement_marker = initial_verification.marker
        record.final_answer = initial_verification.final_answer
        # The final closed-set solution is filled by _run_offload_turns and is also
        # used as the discussion's current draft.
        record.solution = None
        record.response_state = "draft"
        agent_messages.append(record)

        selected_solution = generation.text
        if self.cache_mode == CACHE_MODE_FREE:
            transcript, selected_solution = self._run_free_turns(
                transcript=transcript,
                initial_generation=generation,
                agent_messages=agent_messages,
                turns=turns,
                context=context,
                question=question,
                example_index=example_index,
                initial_pretranslation_sec=initial_pretranslation_sec,
            )
        else:
            transcript, selected_solution = self._run_retain_turns(
                transcript=transcript,
                initial_generation=generation,
                agent_messages=agent_messages,
                turns=turns,
                context=context,
                question=question,
                example_index=example_index,
                initial_pretranslation_sec=initial_pretranslation_sec,
            )

        prediction = self.extract_final_answer(transcript, selected_solution)
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
            agent_messages=agent_messages,
            turns=turns,
            profile={},
            agent_ids=list(self.node_ids),
            hub_agent_id=self.hub_agent.node_id,
            personas=dict(self.agent_personas),
            cache_mode=self.cache_mode,
        )

    def _agent_message_record(
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
    ) -> AgentMessageRecord:
        agent = self.agents[generation.agent_id]
        return AgentMessageRecord(
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
        turn_ttft_sec = [
            float(message.ttft_sec)
            for message in result.agent_messages
            if message.ttft_sec is not None
        ]
        example_ttft_sec = (
            sum(turn_ttft_sec) / len(turn_ttft_sec) if turn_ttft_sec else None
        )
        result.profile = {
            "latency_sec": float(latency_sec),
            "turn_ttft_sec": turn_ttft_sec,
            "example_ttft_sec": example_ttft_sec,
            "ttft_includes_offload_translation": True,
            "ttft_excludes_verification_retries": True,
            "tokens": len(result.agent_messages) * max(1, self.generation_max_new_tokens),
            "num_agent_messages": len(result.agent_messages),
            "completed_turns": len(result.turns),
            "requested_max_turns": self.max_turns,
            "persona_generator": "expert",
            "response_generator": "simple",
            "discussion_paradigm": "memory",
            "decision_protocol": "turn_supermajority_then_majority_vote",
            "consensus_requires_full_initial_participation": True,
            "supermajority_threshold": MALLM_SUPERMAJORITY_THRESHOLD,
            "supermajority_comparison": ">",
            "verification_retry_policy": "unbounded",
            "verification_retry_count": sum(message.verification_attempts for message in result.agent_messages),
            "verification_failure_count": sum(1 for message in result.agent_messages if message.verification_passed is False),
            "consensus_reached": self._consensus_reached,
            "consensus_turn": self._consensus_turn,
            "consensus_answer": self._consensus_answer,
            "final_decision_method": self._final_decision_method,
            "final_decision_answer": self._final_decision_answer,
            "turns": [
                {
                    "turn_index": record.turn_index,
                    "agent_id": record.agent_id,
                    "agent_votes": dict(record.agent_votes),
                    "vote_counts": dict(record.vote_counts),
                    "consensus_reached": record.consensus_reached,
                    "consensus_answer": record.consensus_answer,
                }
                for record in result.turns
            ],
            "final_agent_votes": dict(self._final_agent_votes),
            "final_vote_counts": {
                "Yes": sum(v == "Yes" for v in self._final_agent_votes.values()),
                "No": sum(v == "No" for v in self._final_agent_votes.values()),
            },
            "model_memory_gib": None if peak_memory is None else peak_memory.model_bytes / (1024 ** 3),
            "translator_memory_gib": None if peak_memory is None else peak_memory.translator_bytes / (1024 ** 3),
            "kv_memory_gib": self._peak_kv_cache_bytes / (1024 ** 3),
            "kv_cache_memory_samples_gib": [
                sample / (1024 ** 3) for sample in self._kv_cache_memory_samples_bytes
            ],
        }
        return result



__all__ = [
    "CACHE_MODE_FREE",
    "CACHE_MODE_RETAIN",
    "SUPPORTED_CACHE_MODES",
    "AgentRunner",
    "AgentRunnerConfig",
    "AgentRunnerResult",
    "AgentMessageRecord",
    "AgentTurnRecord",
    "KVCacheTranslationAdapter",
]
