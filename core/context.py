from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from .channel_manager import ChannelManager
from .config import Config
from .model_manager import ModelManager
from .topology import Edge, Node


if TYPE_CHECKING:
    from .channel_profiler import ChannelProfiler


@dataclass
class Context:
    config: Config
    nodes: List[Node]
    edges: List[Edge]
    mm: ModelManager
    cm: ChannelManager
    cp: Optional["ChannelProfiler"] = None
