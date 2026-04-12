from typing import Dict, Optional

from transformers import PreTrainedModel

from .model_spec import ModelSpec


def get_model_spec(model: PreTrainedModel) -> ModelSpec:
    config = model.config
    try:
        num_heads = config.n_head
        hidden_size = config.n_embd
        num_layers = config.n_layer
    except AttributeError as exc:
        raise ValueError("This example expects GPT-2 style configs with n_head/n_embd/n_layer.") from exc
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads.")
    model_id = config._name_or_path if hasattr(config, "_name_or_path") else "unknown"
    return ModelSpec(
        model_id=model_id,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads,
    )


class ModelManager:
    def __init__(
        self,
        models: Dict[str, PreTrainedModel],
        model_specs: Optional[Dict[str, ModelSpec]] = None,
    ) -> None:
        self._models = dict(models)
        if model_specs is None:
            self._model_specs = {
                model_id: get_model_spec(model)
                for model_id, model in self._models.items()
            }
        else:
            self._model_specs = dict(model_specs)

    def get_model(self, model_id: str) -> PreTrainedModel:
        return self._models[model_id]

    def get_model_spec(self, model_id: str) -> ModelSpec:
        return self._model_specs[model_id]
