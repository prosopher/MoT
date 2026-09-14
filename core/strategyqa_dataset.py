from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import List, Optional
from urllib.error import URLError
from urllib.request import urlopen


STRATEGYQA_DEFAULT_DATA_DIR = "./strategyqa"
STRATEGYQA_DEFAULT_SPLIT = "dev"
STRATEGYQA_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/{split}.json"
)
SUPPORTED_STRATEGYQA_SPLITS = ("train", "dev")


@dataclass(frozen=True)
class StrategyQAExample:
    id: str
    question: str
    answer: bool
    facts: List[str]

    @property
    def answers(self) -> List[str]:
        return ["yes" if self.answer else "no"]


def _normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in SUPPORTED_STRATEGYQA_SPLITS:
        raise ValueError(
            f"Unsupported StrategyQA split={split!r}; expected one of {SUPPORTED_STRATEGYQA_SPLITS}"
        )
    return normalized


def _strategyqa_path(data_dir: str, split: str) -> Path:
    return Path(data_dir) / f"{_normalize_split(split)}.json"


def ensure_strategyqa_file(
    *,
    data_dir: str = STRATEGYQA_DEFAULT_DATA_DIR,
    split: str = STRATEGYQA_DEFAULT_SPLIT,
) -> Path:
    split = _normalize_split(split)
    path = _strategyqa_path(data_dir, split)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = STRATEGYQA_URL_TEMPLATE.format(split=split)
    try:
        with urlopen(url) as response, path.open("wb") as output:
            output.write(response.read())
    except (OSError, URLError) as exc:
        raise RuntimeError(
            f"Could not download StrategyQA {split!r} split from {url}. "
            f"Download that JSON manually and save it as {path}."
        ) from exc
    return path


def load_strategyqa_examples(
    *,
    data_dir: str = STRATEGYQA_DEFAULT_DATA_DIR,
    split: str = STRATEGYQA_DEFAULT_SPLIT,
    max_examples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[StrategyQAExample]:
    path = ensure_strategyqa_file(data_dir=data_dir, split=split)
    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    examples = [
        StrategyQAExample(
            id=str(row["qid"]),
            question=str(row["question"]).strip(),
            answer=bool(row["answer"]),
            facts=[str(fact).strip() for fact in row.get("facts", []) if str(fact).strip()],
        )
        for row in rows
    ]
    if shuffle:
        random.Random(int(seed)).shuffle(examples)
    if max_examples is not None:
        examples = examples[: max(0, int(max_examples))]
    return examples


__all__ = [
    "STRATEGYQA_DEFAULT_DATA_DIR",
    "STRATEGYQA_DEFAULT_SPLIT",
    "STRATEGYQA_URL_TEMPLATE",
    "StrategyQAExample",
    "ensure_strategyqa_file",
    "load_strategyqa_examples",
]
