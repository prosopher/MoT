from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class ModelSpec:
    model_id: str
    num_layers: int
    hidden_size: int
    num_heads: int
    head_dim: int


_NUM_HEAD_FIELDS = ("num_attention_heads", "n_head")
_HIDDEN_SIZE_FIELDS = ("hidden_size", "n_embd", "d_model")
_NUM_LAYERS_FIELDS = ("num_hidden_layers", "n_layer", "n_layers")


def _read_required_config_value(config: Any, field_names: Iterable[str], label: str) -> int:
    for field_name in field_names:
        value = getattr(config, field_name, None)
        if value is not None:
            return int(value)
    supported = ", ".join(field_names)
    raise ValueError(f"Could not infer {label}; expected one of: {supported}")


def infer_model_spec_from_config(config: Any, *, default_model_id: str = "unknown") -> ModelSpec:
    num_heads = _read_required_config_value(config, _NUM_HEAD_FIELDS, "num_heads")
    hidden_size = _read_required_config_value(config, _HIDDEN_SIZE_FIELDS, "hidden_size")
    num_layers = _read_required_config_value(config, _NUM_LAYERS_FIELDS, "num_layers")

    if hidden_size % num_heads != 0:
        raise ValueError(f"hidden_size must be divisible by num_heads, got {hidden_size} and {num_heads}")

    model_id = getattr(config, "_name_or_path", default_model_id)
    return ModelSpec(
        model_id=model_id,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads,
    )
