from typing import Dict, Hashable

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .model_spec import ModelSpec, infer_model_spec_from_config


class ModelManager:
    def __init__(
        self,
        models: Dict[str, PreTrainedModel],
        tokenizers: Dict[str, PreTrainedTokenizerBase],
    ) -> None:
        self._models: Dict[str, PreTrainedModel] = {}
        self._tokenizers: Dict[str, PreTrainedTokenizerBase] = {}

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
