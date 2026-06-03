from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from .channel_manager import ChannelManager
from .config import Config
from .topology import Edge, Node
from .translator_pool import TranslatorPool


if TYPE_CHECKING:
    from .channel_profiler import ChannelProfiler


@dataclass
class Context:
    config: Config
    nodes: List[Node]
    edges: List[Edge]
    tp: TranslatorPool
    cm: ChannelManager
    cp: Optional["ChannelProfiler"] = None

    @property
    def mm(self) -> TranslatorPool:
        return self.tp

    @mm.setter
    def mm(self, value: TranslatorPool) -> None:
        self.tp = value
