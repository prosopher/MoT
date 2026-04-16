from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .common import read_json, write_json
from .topology import Edge


@dataclass(frozen=True)
class Channel:
    src_layer_idx: int
    dst_layer_idx: int




def build_resolved_channels_path(output_dir: Path) -> Path:
    return Path(output_dir) / "resolved_channels.json"


def save_resolved_channels(output_dir: Path, cm: "ChannelManager", edges: List[Edge]) -> Path:
    output_path = build_resolved_channels_path(Path(output_dir))
    payload = {
        edge.id: [
            {"src_layer_idx": channel.src_layer_idx, "dst_layer_idx": channel.dst_layer_idx}
            for channel in cm.get_channels(edge.id)
        ]
        for edge in edges
    }
    write_json(str(output_path), payload)
    return output_path


def load_resolved_channels(config_path: Path, cm: "ChannelManager", edges: List[Edge]) -> None:
    payload = read_json(str(config_path))
    for edge in edges:
        channels = payload.get(edge.id)
        if channels is None:
            raise KeyError(f"Missing resolved channels for edge={edge.id} in {config_path}")
        for channel in channels:
            cm.add_channel(
                edge.id,
                src_layer_idx=channel["src_layer_idx"],
                dst_layer_idx=channel["dst_layer_idx"],
            )


def has_resolved_channels(cm: "ChannelManager", edges: List[Edge]) -> bool:
    return all(len(cm.get_channels(edge.id)) > 0 for edge in edges)

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
