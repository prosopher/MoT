from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from .channel_manager import ChannelManager
from .config import Config
from .topology import Edge, Node, build_nodes_and_edges
from .translator_pool import TranslatorPool


if TYPE_CHECKING:
    from .channel_profiler import ChannelProfiler


@dataclass
class Context:
    config: Config
    cp: Optional["ChannelProfiler"] = None
    nodes: List[Node] = field(init=False)
    edges: List[Edge] = field(init=False)
    tp: TranslatorPool = field(init=False)
    cm: ChannelManager = field(init=False)

    def __post_init__(self) -> None:
        self.nodes, self.edges = build_nodes_and_edges(
            self.config.model_ids,
            self.config.model_directions,
        )
        self.tp = TranslatorPool(self.config, self.nodes)
        self.cm = ChannelManager(self.edges)
