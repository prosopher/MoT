from typing import Dict

from transformers import PreTrainedModel

from .model_spec import ModelSpec, infer_model_spec_from_config


class ModelManager:
    def __init__(
        self,
        models: Dict[str, PreTrainedModel],
    ) -> None:
        self._models = dict(models)
        self._model_specs = {
            node_id: infer_model_spec_from_config(model.config)
            for node_id, model in self._models.items()
        }

    def get_model(self, node_id: str) -> PreTrainedModel:
        return self._models[node_id]

    def get_model_spec(self, node_id: str) -> ModelSpec:
        return self._model_specs[node_id]
