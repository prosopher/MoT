from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreTrainedTokenizerBase:
    pass


class TinyTokenizer(PreTrainedTokenizerBase):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"
        self.model_max_length = 128
        self._vocab_size = 128

    def _encode_text(self, text: str) -> List[int]:
        if not text:
            return [self.eos_token_id]
        return [2 + (ord(ch) % (self._vocab_size - 2)) for ch in text]

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        verbose: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **_: object,
    ):
        del verbose

        token_ids = self._encode_text(text)
        if add_special_tokens:
            token_ids = token_ids + [self.eos_token_id]

        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]

        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([token_ids], dtype=torch.long))
        return SimpleNamespace(input_ids=token_ids)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {self.pad_token_id, self.eos_token_id}:
                continue
            chars.append(chr((int(token_id) - 2) % 95 + 32))
        return "".join(chars)


class AutoTokenizer:
    @staticmethod
    def from_pretrained(model_id: str) -> TinyTokenizer:
        return TinyTokenizer(model_id)


class AutoConfig:
    @staticmethod
    def from_pretrained(model_id: str) -> "TinyConfig":
        return TinyConfig(_name_or_path=model_id)


class PreTrainedModel(nn.Module):
    config: object


@dataclass
class TinyConfig:
    _name_or_path: str
    n_head: int = 2
    n_embd: int = 8
    n_layer: int = 2
    vocab_size: int = 128
    n_positions: int = 2048


class TinyGPT2Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.split_size = hidden_size
        self.reorder_and_upcast_attn = False
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.resid_dropout = nn.Identity()

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = tensor.shape
        return tensor.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, num_heads * head_dim)

    def _attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask=None,
        head_mask=None,
    ):
        del attention_mask, head_mask
        scale = 1.0 / math.sqrt(float(self.head_dim))
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache: bool = True,
        output_attentions: bool = False,
    ):
        del encoder_hidden_states, encoder_attention_mask
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        attn_output, attn_weights = self._attn(
            query,
            key,
            value,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output,)
        if use_cache:
            outputs = outputs + ((key, value),)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class TinyGPT2MLP(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc_out = nn.Linear(4 * hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.gelu(self.fc_in(hidden_states)))


class TinyGPT2Block(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = TinyGPT2Attention(hidden_size, num_heads)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = TinyGPT2MLP(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache: bool = True,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        hidden_states = residual + attn_output
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + (attn_outputs[1],)
            if output_attentions and len(attn_outputs) > 2:
                outputs = outputs + (attn_outputs[2],)
        elif output_attentions and len(attn_outputs) > 1:
            outputs = outputs + (attn_outputs[1],)
        return outputs


class TinyTransformer(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Identity()
        self.h = nn.ModuleList([TinyGPT2Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)


class TinyCausalLM(PreTrainedModel):
    def __init__(self, model_id: str, torch_dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.config = TinyConfig(_name_or_path=model_id)

        seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(model_id)) % 10_000
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.transformer = TinyTransformer(self.config)
            self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        self.to(dtype=torch_dtype)

    def forward(self, input_ids: torch.Tensor, past_key_values=None, use_cache: bool = True):
        batch_size, seq_len = input_ids.shape
        past_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_length = int(past_key_values[0][0].shape[2])

        position_ids = torch.arange(
            past_length,
            past_length + seq_len,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        hidden_states = self.transformer.drop(hidden_states)

        presents = []
        for layer_idx, block in enumerate(self.transformer.h):
            layer_past = None if past_key_values is None else past_key_values[layer_idx]
            block_outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=use_cache,
                output_attentions=False,
            )
            hidden_states = block_outputs[0]
            if use_cache:
                presents.append(block_outputs[1])

        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(
            logits=logits,
            past_key_values=tuple(presents) if use_cache else None,
            last_hidden_state=hidden_states,
        )


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(model_id: str, torch_dtype: torch.dtype = torch.float32) -> TinyCausalLM:
        return TinyCausalLM(model_id=model_id, torch_dtype=torch_dtype)
