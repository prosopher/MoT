from dataclasses import dataclass
from typing import List

from transformers import PreTrainedTokenizerBase

from .config import Config
from .model_manager import ModelManager
from .topology import Edge, Node


@dataclass
class Context:
    config: Config
    nodes: List[Node]
    edges: List[Edge]
    mm: ModelManager
    tokenizer: PreTrainedTokenizerBase
