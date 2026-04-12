from dataclasses import dataclass
from typing import Dict

from .config import Config
from .model_spec import ModelSpec


@dataclass
class Context:
    config: Config
    model_specs: Dict[str, ModelSpec]
