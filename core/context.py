from dataclasses import dataclass
from typing import Dict, List

from .config import Config
from .model_spec import ModelSpec
from .topology import Edge, Node


@dataclass
class Context:
    config: Config
    model_specs: Dict[str, ModelSpec]
    nodes: List[Node]
    edges: List[Edge]
