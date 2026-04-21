from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .common import PastKeyValues, extract_past_key_values
from .eval_util import append_input_ids_to_past, postprocess_generated_answer


@dataclass
class GenerationOutput:
    text: str
    generated_token_ids: List[int]
    final_past_key_values: PastKeyValues
    generated_tokens: int


class Agent:
    def __init__(
        self,
        *,
        node_id: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
    ) -> None:
        self.node_id = node_id
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def encode_text(self, text: str) -> List[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def to_tensor(self, token_ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def build_past_from_token_ids(self, token_ids: Sequence[int]) -> PastKeyValues:
        ids = list(token_ids)
        if len(ids) < 1:
            raise ValueError(f"{self.node_id}: token_ids must contain at least one token.")
        return extract_past_key_values(self.model, self.to_tensor(ids))

    @torch.inference_mode()
    def append_token_ids_to_past(
        self,
        past_key_values: PastKeyValues,
        token_ids: Sequence[int],
    ) -> PastKeyValues:
        ids = list(token_ids)
        if not ids:
            return past_key_values
        return append_input_ids_to_past(
            model=self.model,
            past_key_values=past_key_values,
            input_ids=self.to_tensor(ids),
        )

    @torch.inference_mode()
    def generate_from_past_and_seed(
        self,
        *,
        past_key_values: PastKeyValues,
        seed_token_id: int,
        max_new_tokens: int,
    ) -> GenerationOutput:
        generated_token_ids: List[int] = []
        current_input_ids = torch.tensor([[seed_token_id]], dtype=torch.long, device=self.device)
        current_past = past_key_values
        eos_token_id = self.tokenizer.eos_token_id
        pending_input_needs_cache = True

        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids=current_input_ids,
                past_key_values=current_past,
                use_cache=True,
            )
            current_past = outputs.past_key_values
            pending_input_needs_cache = False

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token_id = int(next_token.item())
            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            generated_token_ids.append(next_token_id)
            current_input_ids = next_token
            pending_input_needs_cache = True

        if pending_input_needs_cache:
            finalize_outputs = self.model(
                input_ids=current_input_ids,
                past_key_values=current_past,
                use_cache=True,
            )
            current_past = finalize_outputs.past_key_values

        decoded = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return GenerationOutput(
            text=postprocess_generated_answer(decoded),
            generated_token_ids=generated_token_ids,
            final_past_key_values=current_past,
            generated_tokens=len(generated_token_ids),
        )

    def debug(self, label: str, output: GenerationOutput, print_fn: Callable[[str], None] = print) -> None:
        print_fn(
            f"[{self.node_id}] {label} | raw={output.text!r} | generated_tokens={output.generated_tokens}"
        )


class HubAgent(Agent):
    pass


def measure_kv_bytes(past_key_values: Optional[PastKeyValues]) -> int:
    if past_key_values is None:
        return 0
    total = 0
    for key, value in past_key_values:
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return int(total)


def sum_live_kv_bytes(past_map: Dict[str, Optional[PastKeyValues]]) -> int:
    total = 0
    seen_object_ids = set()
    for past in past_map.values():
        if past is None:
            continue
        object_id = id(past)
        if object_id in seen_object_ids:
            continue
        seen_object_ids.add(object_id)
        total += measure_kv_bytes(past)
    return int(total)
