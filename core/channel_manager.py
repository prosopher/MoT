from dataclasses import dataclass
from typing import Dict, List

from .topology import Edge


@dataclass(frozen=True)
class Channel:
    src_layer_idx: int
    dst_layer_idx: int


class ChannelManager:
    def __init__(self, edges: List[Edge]) -> None:
        self._channel_map: Dict[str, List[Channel]] = {edge.id: [] for edge in edges}

    def add_channel(self, edge_id: str, src_layer_idx: int, dst_layer_idx: int) -> None:
        if src_layer_idx < 0 or dst_layer_idx < 0:
            raise ValueError(
                f"Channels for edge={edge_id} must use non-negative layer indices: "
                f"new=({src_layer_idx}, {dst_layer_idx})"
            )

        channels = self._channel_map[edge_id]
        if channels:
            last_channel = channels[-1]
            if src_layer_idx <= last_channel.src_layer_idx or dst_layer_idx <= last_channel.dst_layer_idx:
                raise ValueError(
                    f"Channels for edge={edge_id} must be appended in strictly increasing layer order: "
                    f"last=({last_channel.src_layer_idx}, {last_channel.dst_layer_idx}), "
                    f"new=({src_layer_idx}, {dst_layer_idx})"
                )
        channels.append(Channel(src_layer_idx, dst_layer_idx))

    def get_channels(self, edge_id: str) -> List[Channel]:
        return self._channel_map[edge_id]

    def get_src_layer_indices(self, edge_id: str) -> List[int]:
        return [channel.src_layer_idx for channel in self.get_channels(edge_id)]

    def get_tgt_layer_indices(self, edge_id: str) -> List[int]:
        return [channel.dst_layer_idx for channel in self.get_channels(edge_id)]

    def get_src_layer_start_idx(self, edge_id: str) -> int:
        channels = self.get_channels(edge_id)
        return channels[0].src_layer_idx

    def get_src_layer_end_idx(self, edge_id: str) -> int:
        channels = self.get_channels(edge_id)
        return channels[-1].src_layer_idx

    def get_tgt_layer_start_idx(self, edge_id: str) -> int:
        channels = self.get_channels(edge_id)
        return channels[0].dst_layer_idx

    def get_tgt_layer_end_idx(self, edge_id: str) -> int:
        channels = self.get_channels(edge_id)
        return channels[-1].dst_layer_idx
