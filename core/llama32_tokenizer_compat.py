"""Llama 3.x tokenizer compatibility for transformers==4.35.2.

Llama 3 tokenizers are byte-level BPE tokenizers derived from tiktoken.  Newer
``tokenizer.json`` files use the BPE ``ignore_merges`` option, which is not
understood by the tokenizers versions allowed by transformers 4.35.2.  This
module provides a small Python tokenizer fallback that preserves those
semantics instead of deleting ``ignore_merges`` and silently changing token
IDs.

The implementation intentionally targets the Llama 3/3.1/3.2 tokenizer JSON
shape used by the official checkpoints: regex split -> ByteLevel -> BPE with
``ignore_merges=true`` plus added special tokens.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import regex
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import AddedToken
from transformers.utils import cached_file


_LLAMA3_PATTERN = regex.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _bytes_to_unicode() -> Dict[int, str]:
    """GPT-2/Llama-3 reversible byte-to-unicode mapping used by ByteLevel."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = list(bs)
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(codepoint) for codepoint in cs)))


def _get_pairs(symbols: Sequence[str]) -> set[Tuple[str, str]]:
    if len(symbols) < 2:
        return set()
    return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}


def _added_token_from_dict(token_dict: Dict[str, object]) -> AddedToken:
    return AddedToken(
        str(token_dict["content"]),
        single_word=bool(token_dict.get("single_word", False)),
        lstrip=bool(token_dict.get("lstrip", False)),
        rstrip=bool(token_dict.get("rstrip", False)),
        normalized=bool(token_dict.get("normalized", False)),
        special=bool(token_dict.get("special", False)),
    )


def _token_content(value):
    if isinstance(value, dict):
        return value.get("content")
    if value is None:
        return None
    return str(value)


class Llama32TokenizerCompat(PreTrainedTokenizer):
    """Slow-but-exact Llama 3 byte-level BPE fallback for old tokenizers."""

    vocab_files_names = {"tokenizer_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        *,
        tokenizer_file: str,
        tokenizer_config: Optional[Dict[str, object]] = None,
        **kwargs,
    ) -> None:
        self.tokenizer_file = tokenizer_file
        with open(tokenizer_file, "r", encoding="utf-8") as handle:
            tokenizer_json = json.load(handle)

        model = tokenizer_json.get("model")
        if not isinstance(model, dict) or model.get("type") != "BPE":
            raise ValueError("Llama 3 compatibility expects a BPE tokenizer.json model.")
        if not bool(model.get("ignore_merges", False)):
            raise ValueError(
                "Llama 3 compatibility expects tokenizer.json model.ignore_merges=true."
            )

        vocab = model.get("vocab")
        if not isinstance(vocab, dict) or not vocab:
            raise ValueError("Llama 3 tokenizer.json is missing its BPE vocabulary.")
        self.encoder: Dict[str, int] = {str(token): int(index) for token, index in vocab.items()}
        self.decoder: Dict[int, str] = {index: token for token, index in self.encoder.items()}

        merges = model.get("merges") or []
        parsed_merges: List[Tuple[str, str]] = []
        for merge in merges:
            if isinstance(merge, str):
                parts = merge.split(" ")
                if len(parts) != 2:
                    raise ValueError(f"Unexpected BPE merge entry: {merge!r}")
                parsed_merges.append((parts[0], parts[1]))
            elif isinstance(merge, (list, tuple)) and len(merge) == 2:
                parsed_merges.append((str(merge[0]), str(merge[1])))
            else:
                raise ValueError(f"Unexpected BPE merge entry: {merge!r}")
        self.bpe_ranks = {pair: rank for rank, pair in enumerate(parsed_merges)}

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        self.errors = "replace"

        config = dict(tokenizer_config or {})
        added_tokens_decoder: Dict[int, AddedToken] = {}

        configured_added = config.get("added_tokens_decoder")
        if isinstance(configured_added, dict):
            for token_id, token_info in configured_added.items():
                if isinstance(token_info, dict) and "content" in token_info:
                    added_tokens_decoder[int(token_id)] = _added_token_from_dict(token_info)

        # Some mirrors keep the metadata only in tokenizer.json.  Fill any
        # missing entries from there while preserving the checkpoint IDs.
        for token_info in tokenizer_json.get("added_tokens") or []:
            if not isinstance(token_info, dict) or "id" not in token_info or "content" not in token_info:
                continue
            token_id = int(token_info["id"])
            if token_id not in added_tokens_decoder:
                added_tokens_decoder[token_id] = _added_token_from_dict(token_info)

        bos_token = _token_content(config.get("bos_token")) or "<|begin_of_text|>"
        eos_token = _token_content(config.get("eos_token")) or "<|end_of_text|>"
        pad_token = _token_content(config.get("pad_token"))
        unk_token = _token_content(config.get("unk_token"))

        named_specials = {token for token in (bos_token, eos_token, pad_token, unk_token) if token}
        additional_special_tokens: List[str] = []
        configured_additional = config.get("additional_special_tokens")
        if isinstance(configured_additional, list):
            for token in configured_additional:
                content = _token_content(token)
                if content and content not in named_specials and content not in additional_special_tokens:
                    additional_special_tokens.append(content)
        for token_id in sorted(added_tokens_decoder):
            token = added_tokens_decoder[token_id]
            if token.special and token.content not in named_specials and token.content not in additional_special_tokens:
                additional_special_tokens.append(token.content)

        self.add_bos_token = bool(config.get("add_bos_token", True))
        self.add_eos_token = bool(config.get("add_eos_token", False))

        base_kwargs = {
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "unk_token": unk_token,
            "additional_special_tokens": additional_special_tokens,
            "added_tokens_decoder": added_tokens_decoder,
            "model_max_length": int(config.get("model_max_length", 131072)),
            "clean_up_tokenization_spaces": bool(config.get("clean_up_tokenization_spaces", False)),
            "add_bos_token": self.add_bos_token,
            "add_eos_token": self.add_eos_token,
            # Byte-level decoding must not inject spaces around special tokens.
            "spaces_between_special_tokens": False,
        }
        if config.get("chat_template") is not None:
            base_kwargs["chat_template"] = config["chat_template"]

        # Preserve harmless tokenizer configuration values that the 4.35 base
        # class understands, while excluding loader/backend metadata.
        ignored_config_keys = {
            "added_tokens_decoder",
            "additional_special_tokens",
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "tokenizer_class",
            "tokenizer_file",
            "backend",
            "is_local",
            "model_max_length",
            "clean_up_tokenization_spaces",
            "add_bos_token",
            "add_eos_token",
            "chat_template",
        }
        for key, value in config.items():
            if key not in ignored_config_keys and key not in base_kwargs:
                base_kwargs[key] = value
        base_kwargs.update(kwargs)
        super().__init__(**base_kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def get_vocab(self) -> Dict[str, int]:
        vocab = dict(self.encoder)
        vocab.update(getattr(self, "_added_tokens_encoder", {}))
        return vocab

    @lru_cache(maxsize=65536)
    def _bpe(self, token: str) -> Tuple[str, ...]:
        # This early return is the important Llama 3/tiktoken semantic that
        # tokenizers<0.19 cannot represent: a complete vocab token wins even
        # when no merge path would construct it.
        if token in self.encoder:
            return (token,)

        word: Tuple[str, ...] = tuple(token)
        if len(word) <= 1:
            return word

        while True:
            pairs = _get_pairs(word)
            if not pairs:
                break
            best_pair = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if best_pair not in self.bpe_ranks:
                break

            first, second = best_pair
            merged: List[str] = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_index])
                index = next_index
                if index < len(word) - 1 and word[index] == first and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
        return word

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        tokens: List[str] = []
        for piece in _LLAMA3_PATTERN.findall(text):
            byte_piece = "".join(self.byte_encoder[byte] for byte in piece.encode("utf-8"))
            tokens.extend(self._bpe(byte_piece))
        return tokens

    def _convert_token_to_id(self, token: str) -> Optional[int]:
        return self.encoder.get(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, "")

    def _decode_byte_tokens(self, tokens: Iterable[str]) -> str:
        encoded = "".join(tokens)
        try:
            raw = bytearray(self.byte_decoder[char] for char in encoded)
        except KeyError as error:
            raise ValueError(f"Invalid byte-level token while decoding Llama 3: {error.args[0]!r}") from error
        return raw.decode("utf-8", errors=self.errors)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # Keep added/special tokens literal and byte-decode only BPE tokens.
        output: List[str] = []
        byte_tokens: List[str] = []
        added = getattr(self, "_added_tokens_encoder", {})
        for token in tokens:
            if token in added:
                if byte_tokens:
                    output.append(self._decode_byte_tokens(byte_tokens))
                    byte_tokens = []
                output.append(token)
            else:
                byte_tokens.append(token)
        if byte_tokens:
            output.append(self._decode_byte_tokens(byte_tokens))
        return "".join(output)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        bos = [self.bos_token_id] if self.add_bos_token and self.bos_token_id is not None else []
        eos = [self.eos_token_id] if self.add_eos_token and self.eos_token_id is not None else []
        output = bos + token_ids_0 + eos
        if token_ids_1 is not None:
            output += bos + token_ids_1 + eos
        return output

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        bos = [1] if self.add_bos_token and self.bos_token_id is not None else []
        eos = [1] if self.add_eos_token and self.eos_token_id is not None else []
        mask = bos + ([0] * len(token_ids_0)) + eos
        if token_ids_1 is not None:
            mask += bos + ([0] * len(token_ids_1)) + eos
        return mask

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        first_len = len(self.build_inputs_with_special_tokens(token_ids_0))
        output = [0] * first_len
        if token_ids_1 is not None:
            second_len = len(self.build_inputs_with_special_tokens(token_ids_1))
            output += [1] * second_len
        return output

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            return ()
        filename = ((filename_prefix + "-") if filename_prefix else "") + "tokenizer.json"
        output_path = os.path.join(save_directory, filename)
        if os.path.abspath(output_path) != os.path.abspath(self.tokenizer_file):
            with open(self.tokenizer_file, "rb") as source, open(output_path, "wb") as target:
                target.write(source.read())
        return (output_path,)


def _load_json_if_available(model_id: str, filename: str) -> Dict[str, object]:
    try:
        path = cached_file(model_id, filename)
    except Exception:
        return {}
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def load_llama32_tokenizer_compat(model_id: str) -> Llama32TokenizerCompat:
    tokenizer_path = cached_file(model_id, "tokenizer.json")
    if tokenizer_path is None:
        raise FileNotFoundError(f"tokenizer.json was not found for {model_id}")

    tokenizer_config = _load_json_if_available(model_id, "tokenizer_config.json")
    special_tokens_map = _load_json_if_available(model_id, "special_tokens_map.json")
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        if key not in tokenizer_config and key in special_tokens_map:
            tokenizer_config[key] = special_tokens_map[key]

    return Llama32TokenizerCompat(
        tokenizer_file=tokenizer_path,
        tokenizer_config=tokenizer_config,
    )
