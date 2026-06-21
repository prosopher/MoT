from typing import Any, Dict, Iterable, Iterator

import torch
import torch.nn as nn
from .common import load_frozen_model, load_tokenizer
from .model import Model
from .model_spec import ModelSpec, infer_model_spec_from_config
from .topology import Node


class TranslatorPool:
    def __init__(self, config: Any, nodes: Iterable[Node]) -> None:
        self.models: Dict[str, Model] = {}
        self.translators: Dict[str, nn.Module] = {}

        unique_models: Dict[str, Model] = {}
        for node in nodes:
            if node.model_id not in unique_models:
                model = load_frozen_model(node.model_id, device=config.device, dtype=config.dtype)
                tokenizer = load_tokenizer(node.model_id)
                unique_models[node.model_id] = Model(node.model_id, model, tokenizer)
            self.models[node.id] = unique_models[node.model_id]

        self._model_specs = {
            node_id: infer_model_spec_from_config(model.config, default_model_id=model.id)
            for node_id, model in self.models.items()
        }

    def get_model(self, node_id: str) -> Model:
        return self.models[node_id]

    def get_model_spec(self, node_id: str) -> ModelSpec:
        return self._model_specs[node_id]

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
