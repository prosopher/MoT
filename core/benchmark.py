from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BenchmarkInfo:
    name: str
    choices: Tuple[str, ...]

    def __post_init__(self) -> None:
        name = self.name.strip()
        choices = tuple(choice.strip() for choice in self.choices)
        if not name:
            raise ValueError("benchmark name must not be empty")
        if len(choices) < 2 or any(not choice for choice in choices):
            raise ValueError("benchmark choices must contain at least two non-empty values")
        normalized = [choice.casefold() for choice in choices]
        if len(set(normalized)) != len(normalized):
            raise ValueError("benchmark choices must be unique ignoring case")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "choices", choices)
