from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from core.model import Model
from core.common import PastKeyValues, TokenIDs, ensure_token_ids_model, extract_past_key_values
from core.eval_util import append_token_ids_to_past


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
        model: Model,
        device: str,
        max_new_tokens: int = 64,
        stop_sequences: Optional[Sequence[str]] = None,
        max_prompt_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> None:
        self.node_id = node_id
        self.model = model
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.stop_sequences = tuple(stop_sequences or ())
        self.max_prompt_tokens = max_prompt_tokens
        self.temperature = float(temperature)
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
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
    ) -> TokenIDs:
        encoded = self.model.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = TokenIDs(encoded.input_ids, model_id=self.model.id)
        token_limit = self.max_prompt_tokens if max_input_tokens is None else max_input_tokens
        if token_limit is not None and token_ids.shape[1] > token_limit:
            if truncation_side == "left":
                token_ids = token_ids[:, -token_limit:]
            elif truncation_side == "right":
                token_ids = token_ids[:, :token_limit]
            else:
                raise ValueError(f"Unsupported truncation_side={truncation_side!r}")
        if token_ids.shape[1] < 1:
            raise ValueError("Agent text must tokenize to at least one token.")
        return token_ids.to(self.device)
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

        prompt_token_ids = prompt_ids[:, :-1]
        seed_token = prompt_ids[:, -1:]
        if self.past_key_values is None:
            prompt_past = extract_past_key_values(self.model, prompt_token_ids)
        else:
            prompt_past = append_token_ids_to_past(
                model=self.model,
                past_key_values=self.past_key_values,
                token_ids=prompt_token_ids,
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

    def _eos_token_ids(self) -> Set[int]:
        """Return every EOS id advertised by the tokenizer/model.

        Some recent model families, including Qwen3, use more than one valid
        end-of-sequence token in ``generation_config``.  The hand-written
        decoding loop cannot rely only on ``tokenizer.eos_token_id`` or it may
        continue generating after another configured EOS token is produced.
        """

        eos_token_ids: Set[int] = set()

        def add_ids(value) -> None:
            if value is None:
                return
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().reshape(-1).tolist()
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    add_ids(item)
                return
            try:
                eos_token_ids.add(int(value))
            except (TypeError, ValueError):
                return

        add_ids(getattr(self.model.tokenizer, "eos_token_id", None))
        add_ids(getattr(getattr(self.model, "config", None), "eos_token_id", None))
        add_ids(getattr(getattr(self.model, "generation_config", None), "eos_token_id", None))
        return eos_token_ids

    def _turn_terminator_token_id(self) -> Optional[int]:
        """Resolve the chat-template assistant turn terminator without model-family branches."""
        tokenizer = self.model.tokenizer
        eos_token_ids = self._eos_token_ids()
        template = getattr(tokenizer, "chat_template", None)
        if isinstance(template, dict):
            template = template.get("default") or next(iter(template.values()), None)

        if isinstance(template, str) and template.strip() and hasattr(tokenizer, "apply_chat_template"):
            marker = "__AGENT_RUNNER_ASSISTANT_END_PROBE__"
            messages = [
                {"role": "user", "content": "probe"},
                {"role": "assistant", "content": marker},
            ]
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    chat_template=template,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                rendered = str(rendered)
                marker_idx = rendered.rfind(marker)
                if marker_idx >= 0:
                    suffix = rendered[marker_idx + len(marker) :]
                    encoded = tokenizer(suffix, add_special_tokens=False)
                    suffix_ids = getattr(encoded, "input_ids", encoded.get("input_ids") if isinstance(encoded, dict) else None)
                    if isinstance(suffix_ids, torch.Tensor):
                        suffix_ids = suffix_ids.detach().cpu().reshape(-1).tolist()
                    elif suffix_ids and isinstance(suffix_ids[0], (list, tuple)):
                        suffix_ids = list(suffix_ids[0])
                    for token_id in suffix_ids or []:
                        token_id = int(token_id)
                        if token_id in eos_token_ids:
                            return token_id
            except Exception as error:
                logging.debug("Could not infer assistant turn terminator from chat template: %s", error)

        # Plain-prompt models do not have a chat-specific terminator.  Their declared
        # tokenizer EOS is the safest generic fallback; if absent, use any configured EOS.
        tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
        if tokenizer_eos is not None:
            try:
                return int(tokenizer_eos)
            except (TypeError, ValueError):
                pass
        return min(eos_token_ids) if eos_token_ids else None

    @torch.inference_mode()
    def generate_response(
        self,
        prompt_text: str,
        *,
        temperature: Optional[float] = None,
    ) -> AgentGeneration:
        effective_temperature = self.temperature if temperature is None else float(temperature)
        if effective_temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {effective_temperature}")
        tokens_before = self.cache_seq_len
        current_past, current_token_ids, tokens_prompt = self._prefill_prompt(prompt_text)
        generated_token_ids: List[int] = []
        terminal_token_id: Optional[int] = None
        eos_token_ids = self._eos_token_ids()
        # A generated token is not added to past_key_values at prediction time;
        # it is added only when it is fed back as input in the next decoding step.
        # Track only the generated token that is still visible but not yet cached.
        uncached_generated_token: Optional[torch.Tensor] = None
        stopped_by_stop_sequence = False

        for _ in range(max(0, self.max_new_tokens)):
            ensure_token_ids_model(self.model, current_token_ids)
            outputs = self.model(
                input_ids=current_token_ids.as_tensor(),
                past_key_values=current_past,
                use_cache=True,
            )
            current_past = outputs.past_key_values
            # The input token for this step has now been incorporated into KV.
            # If it was a previously generated visible token, it no longer needs
            # the final one-token cache append below.
            uncached_generated_token = None

            next_token_logits = outputs.logits[:, -1, :]
            if effective_temperature > 0.0:
                probabilities = torch.softmax(next_token_logits.float() / effective_temperature, dim=-1)
                next_token_tensor = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token_tensor = next_token_logits.argmax(dim=-1, keepdim=True)
            next_token = TokenIDs(next_token_tensor, model_id=current_token_ids.model_id)
            next_token_id = int(next_token.item())

            if next_token_id in eos_token_ids:
                # Keep the model's turn terminator in the resident cache.  For chat
                # models (Qwen/Gemma/Llama and others), EOS is commonly also the
                # assistant end-of-turn token.  The next logical Agent appends a new
                # user turn to this cache, so dropping the terminator would leave the
                # cached conversation structurally incomplete.  It remains excluded
                # from visible/generated_token_ids.
                terminal_token_id = next_token_id
                uncached_generated_token = next_token
                break

            generated_token_ids.append(next_token_id)
            uncached_generated_token = next_token
            decoded_so_far = self.model.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            _, matched_stop = self._trim_at_stop_sequence(decoded_so_far, self.stop_sequences)
            current_token_ids = next_token
            if matched_stop is not None:
                stopped_by_stop_sequence = True
                break

        hit_max_new_tokens = (
            self.max_new_tokens > 0
            and terminal_token_id is None
            and not stopped_by_stop_sequence
            and len(generated_token_ids) >= self.max_new_tokens
        )

        # The final predicted visible token has not entered the KV cache if the
        # loop ended before another decoding step consumed it. This includes an
        # EOS/end-of-turn token predicted by a chat model; it is cached but not
        # exposed as visible completion text.
        if uncached_generated_token is not None:
            ensure_token_ids_model(self.model, uncached_generated_token)
            outputs = self.model(
                input_ids=uncached_generated_token.as_tensor(),
                past_key_values=current_past,
                use_cache=True,
            )
            current_past = outputs.past_key_values

        if hit_max_new_tokens:
            forced_terminal_token_id = self._turn_terminator_token_id()
            if forced_terminal_token_id is None:
                raise RuntimeError(
                    f"Agent {self.node_id} reached max_new_tokens without EOS, but no turn terminator could be resolved."
                )
            terminal_token_id = int(forced_terminal_token_id)
            forced_terminal = TokenIDs(
                torch.tensor([[terminal_token_id]], dtype=torch.long, device=self.device),
                model_id=self.model.id,
            )
            ensure_token_ids_model(self.model, forced_terminal)
            outputs = self.model(
                input_ids=forced_terminal.as_tensor(),
                past_key_values=current_past,
                use_cache=True,
            )
            current_past = outputs.past_key_values

        raw_text = self.model.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        text, _ = self._trim_at_stop_sequence(raw_text, self.stop_sequences)
        text = text.strip()

        prompt_token_ids = self.encode_text(prompt_text).squeeze(0).detach().cpu().tolist()
        expected_cache_token_ids = list(self.cache_token_ids) + prompt_token_ids + list(generated_token_ids)
        if terminal_token_id is not None:
            expected_cache_token_ids.append(terminal_token_id)

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
