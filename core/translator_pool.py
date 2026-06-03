from typing import Dict, Hashable, Iterator

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .model_spec import ModelSpec, infer_model_spec_from_config


class TranslatorPool:
    def __init__(
        self,
        models: Dict[str, PreTrainedModel],
        tokenizers: Dict[str, PreTrainedTokenizerBase],
    ) -> None:
        self._models: Dict[str, PreTrainedModel] = {}
        self._tokenizers: Dict[str, PreTrainedTokenizerBase] = {}
        self.translators: Dict[str, nn.Module] = {}

        unique_models: Dict[Hashable, PreTrainedModel] = {}
        for node_id, model in models.items():
            config = getattr(model, "config", None)
            model_key = getattr(config, "_name_or_path", None) or getattr(model, "name_or_path", None) or id(model)
            if model_key not in unique_models:
                unique_models[model_key] = model
            self._models[node_id] = unique_models[model_key]

        unique_tokenizers: Dict[Hashable, PreTrainedTokenizerBase] = {}
        for node_id, tokenizer in tokenizers.items():
            tokenizer_key = (
                getattr(tokenizer, "name_or_path", None)
                or getattr(tokenizer, "model_id", None)
                or id(tokenizer)
            )
            if tokenizer_key not in unique_tokenizers:
                unique_tokenizers[tokenizer_key] = tokenizer
            self._tokenizers[node_id] = unique_tokenizers[tokenizer_key]

        self._model_specs = {
            node_id: infer_model_spec_from_config(model.config)
            for node_id, model in self._models.items()
        }

    def get_model(self, node_id: str) -> PreTrainedModel:
        return self._models[node_id]

    def get_model_spec(self, node_id: str) -> ModelSpec:
        return self._model_specs[node_id]

    def get_tokenizer(self, node_id: str) -> PreTrainedTokenizerBase:
        return self._tokenizers[node_id]

    def add_translator(self, translator_id: str, translator: nn.Module) -> nn.Module:
        self.translators[translator_id] = translator
        return translator

    def get_translator(self, translator_id: str) -> nn.Module:
        return self.translators[translator_id]

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        for translator in self.translators.values():
            yield from translator.parameters()

    def modules(self) -> Iterator[nn.Module]:
        for translator in self.translators.values():
            yield from translator.modules()

    def to(self, *args, **kwargs) -> "TranslatorPool":
        for translator in self.translators.values():
            translator.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True) -> "TranslatorPool":
        for translator in self.translators.values():
            translator.train(mode)
        return self

    def eval(self) -> "TranslatorPool":
        return self.train(False)
