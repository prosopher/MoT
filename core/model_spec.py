from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass
class ModelSpec:
    model_id: str
    num_layers: int
    hidden_size: int
    num_heads: int
    head_dim: int
    num_key_value_heads: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads
        self.num_key_value_heads = int(self.num_key_value_heads)

    @property
    def kv_hidden_size(self) -> int:
        return int(self.num_key_value_heads) * self.head_dim


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
    configured_num_key_value_heads = getattr(config, "num_key_value_heads", None)
    num_key_value_heads = num_heads if configured_num_key_value_heads is None else int(configured_num_key_value_heads)
    configured_head_dim = getattr(config, "head_dim", None)
    if configured_head_dim is None:
        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads when head_dim is not explicitly configured, "
                f"got hidden_size={hidden_size}, num_heads={num_heads}"
            )
        head_dim = hidden_size // num_heads
    else:
        # Some architectures (notably Qwen3) use an attention projection width
        # num_heads * head_dim that is different from the residual hidden_size.
        # When head_dim is explicit in the model config, it is the authoritative
        # cache/head width and must not be reconstructed from hidden_size.
        head_dim = int(configured_head_dim)
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}")
    if num_key_value_heads < 1:
        raise ValueError(f"num_key_value_heads must be >= 1, got {num_key_value_heads}")
    if num_heads % num_key_value_heads != 0:
        raise ValueError(
            "num_heads must be divisible by num_key_value_heads for GQA/MQA, "
            f"got {num_heads} and {num_key_value_heads}"
        )

    model_id = getattr(config, "_name_or_path", default_model_id)
    return ModelSpec(
        model_id=model_id,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_key_value_heads=num_key_value_heads,
    )
