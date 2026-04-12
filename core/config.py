from dataclasses import dataclass
from typing import Optional

import torch


def resolve_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class Config:
    alg: str
    timestamp: Optional[str]
    output_path: Optional[str]
    device: str

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)
