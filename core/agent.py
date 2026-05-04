from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from core.common import PastKeyValues, extract_past_key_values
from core.eval_util import append_input_ids_to_past


@dataclass
class AgentGeneration:
    agent_id: str
    prompt_text: str
    text: str
    raw_text: str
    generated_token_ids: List[int]
    tokens_before: int
    tokens_after: int
    tokens_prompt: int
    tokens_completion: int

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_token_ids)


def get_past_seq_len(past_key_values: Optional[PastKeyValues]) -> int:
    if past_key_values is None or len(past_key_values) == 0:
        return 0
    key, _ = past_key_values[0]
    return int(key.shape[2])


def slice_past_suffix(past_key_values: PastKeyValues, start_seq_idx: int) -> PastKeyValues:
    start_seq_idx = max(0, int(start_seq_idx))
    return tuple(
        (
            key[:, :, start_seq_idx:, :].contiguous(),
            value[:, :, start_seq_idx:, :].contiguous(),
        )
        for key, value in past_key_values
    )


class Agent:
    """A small stateful wrapper around one causal LM agent and its KV cache."""

    def __init__(
        self,
        *,
        node_id: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        max_new_tokens: int = 64,
        stop_sequences: Optional[Sequence[str]] = None,
        max_prompt_tokens: Optional[int] = None,
    ) -> None:
        self.node_id = node_id
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.stop_sequences = tuple(stop_sequences or ())
        self.max_prompt_tokens = max_prompt_tokens
        self.past_key_values: Optional[PastKeyValues] = None
        # Token ids corresponding 1:1 to the resident KV cache.
        # past_key_values itself does not store the input token ids, so the runner
        # must maintain them explicitly to compute offload deltas without
        # re-tokenizing text or rebuilding source KV at handoff time.
        self.cache_token_ids: List[int] = []
        # Pretranslated target-side KV caches keyed by physical edge id.
        # These are refreshed immediately after this agent's resident cache changes,
        # so offload can slice an already-translated cache without translating at handoff time.
        self.pretranslated_past_by_edge: Dict[str, PastKeyValues] = {}
        self.pretranslated_token_ids_by_edge: Dict[str, List[int]] = {}
        self.model.eval()

    @property
    def cache_seq_len(self) -> int:
        return get_past_seq_len(self.past_key_values)

    def invalidate_pretranslated_caches(self) -> None:
        self.pretranslated_past_by_edge.clear()
        self.pretranslated_token_ids_by_edge.clear()

    def set_pretranslated_cache(
        self,
        *,
        edge_id: str,
        past_key_values: PastKeyValues,
        cache_token_ids: Sequence[int],
    ) -> None:
        token_ids = list(cache_token_ids)
        actual_tokens = get_past_seq_len(past_key_values)
        if len(token_ids) != actual_tokens:
            raise ValueError(
                f"Pretranslated cache/token-id length mismatch for edge {edge_id}: "
                f"token_ids={len(token_ids)} past_tokens={actual_tokens}"
            )
        self.pretranslated_past_by_edge[edge_id] = past_key_values
        self.pretranslated_token_ids_by_edge[edge_id] = token_ids

    def reset(self) -> None:
        self.past_key_values = None
        self.cache_token_ids = []
        self.invalidate_pretranslated_caches()

    def clear_kv_cache(self, *, empty_cuda_cache: bool = False) -> None:
        """Drop this agent's resident KV cache and all derived pretranslated caches."""
        self.past_key_values = None
        self.cache_token_ids = []
        self.invalidate_pretranslated_caches()
        if empty_cuda_cache and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def encode_text(
        self,
        text: str,
        *,
        max_input_tokens: Optional[int] = None,
        truncation_side: str = "left",
    ) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt")
        input_ids = encoded.input_ids
        token_limit = self.max_prompt_tokens if max_input_tokens is None else max_input_tokens
        if token_limit is not None and input_ids.shape[1] > token_limit:
            if truncation_side == "left":
                input_ids = input_ids[:, -token_limit:]
            elif truncation_side == "right":
                input_ids = input_ids[:, :token_limit]
            else:
                raise ValueError(f"Unsupported truncation_side={truncation_side!r}")
        if input_ids.shape[1] < 1:
            raise ValueError("Agent text must tokenize to at least one token.")
        return input_ids.to(self.device)
    def set_replayed_cache(
        self,
        past_key_values: PastKeyValues,
        *,
        cache_token_ids: Sequence[int],
    ) -> None:
        self.past_key_values = past_key_values
        self.cache_token_ids = list(cache_token_ids)
        actual_tokens = self.cache_seq_len
        if len(self.cache_token_ids) != actual_tokens:
            raise ValueError(
                f"Replayed cache/token-id length mismatch for Agent {self.node_id}: "
                f"token_ids={len(self.cache_token_ids)} past_tokens={actual_tokens}"
            )
        self.invalidate_pretranslated_caches()

    @torch.inference_mode()
    def _prefill_prompt(self, prompt_text: str) -> Tuple[Optional[PastKeyValues], torch.Tensor, int]:
        prompt_ids = self.encode_text(prompt_text)
        prompt_tokens = int(prompt_ids.shape[1])
        if prompt_ids.shape[1] == 1:
            return self.past_key_values, prompt_ids, prompt_tokens

        cache_ids = prompt_ids[:, :-1]
        seed_token = prompt_ids[:, -1:]
        if self.past_key_values is None:
            prompt_past = extract_past_key_values(self.model, cache_ids)
        else:
            prompt_past = append_input_ids_to_past(
                model=self.model,
                past_key_values=self.past_key_values,
                input_ids=cache_ids,
            )
        return prompt_past, seed_token, prompt_tokens

    @staticmethod
    def _trim_at_stop_sequence(text: str, stop_sequences: Iterable[str]) -> Tuple[str, Optional[str]]:
        earliest_idx: Optional[int] = None
        matched_stop: Optional[str] = None
        for stop_sequence in stop_sequences:
            if not stop_sequence:
                continue
            idx = text.find(stop_sequence)
            if idx >= 0 and (earliest_idx is None or idx < earliest_idx):
                earliest_idx = idx
                matched_stop = stop_sequence
        if earliest_idx is None:
            return text, None
        return text[:earliest_idx], matched_stop

    @torch.inference_mode()
    def generate_response(self, prompt_text: str) -> AgentGeneration:
        tokens_before = self.cache_seq_len
        current_past, current_input_ids, tokens_prompt = self._prefill_prompt(prompt_text)
        generated_token_ids: List[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        last_generated_token: Optional[torch.Tensor] = None

        for _ in range(max(0, self.max_new_tokens)):
            outputs = self.model(
                input_ids=current_input_ids,
                past_key_values=current_past,
                use_cache=True,
            )
            current_past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token_id = int(next_token.item())

            if eos_token_id is not None and next_token_id == int(eos_token_id):
                break

            generated_token_ids.append(next_token_id)
            last_generated_token = next_token
            decoded_so_far = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            _, matched_stop = self._trim_at_stop_sequence(decoded_so_far, self.stop_sequences)
            current_input_ids = next_token
            if matched_stop is not None:
                break

        # The loop cache contains the token that was fed into the model, not the
        # final predicted token. Feed the last non-EOS generated token once so
        # the stateful cache represents the visible response text as well.
        if last_generated_token is not None:
            outputs = self.model(
                input_ids=last_generated_token,
                past_key_values=current_past,
                use_cache=True,
            )
            current_past = outputs.past_key_values

        raw_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        text, _ = self._trim_at_stop_sequence(raw_text, self.stop_sequences)
        text = text.strip()

        prompt_token_ids = self.encode_text(prompt_text).squeeze(0).detach().cpu().tolist()
        expected_cache_token_ids = list(self.cache_token_ids) + prompt_token_ids + list(generated_token_ids)

        self.past_key_values = current_past
        actual_cache_tokens = self.cache_seq_len
        if len(expected_cache_token_ids) > actual_cache_tokens:
            # This can happen for unusual settings such as max_new_tokens=0, where
            # the final prompt seed token was not fed through the model. Keep the
            # token-id ledger aligned with the actual KV length.
            expected_cache_token_ids = expected_cache_token_ids[:actual_cache_tokens]
        elif len(expected_cache_token_ids) < actual_cache_tokens:
            logging.warning(
                "Agent %s cache token-id ledger shorter than KV cache: token_ids=%d past_tokens=%d",
                self.node_id,
                len(expected_cache_token_ids),
                actual_cache_tokens,
            )
            raise ValueError(
                f"Agent {self.node_id} cache/token-id length mismatch: "
                f"token_ids={len(expected_cache_token_ids)} past_tokens={actual_cache_tokens}"
            )
        self.cache_token_ids = expected_cache_token_ids
        self.invalidate_pretranslated_caches()
        return AgentGeneration(
            agent_id=self.node_id,
            prompt_text=prompt_text,
            text=text,
            raw_text=raw_text,
            generated_token_ids=generated_token_ids,
            tokens_before=tokens_before,
            tokens_after=self.cache_seq_len,
            tokens_prompt=tokens_prompt,
            tokens_completion=len(generated_token_ids),
        )


class HubAgent(Agent):
    """Conversation-starting agent used by AgentRunner."""

    pass
