from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import List, Optional

from datasets import load_dataset

from core.benchmark import BenchmarkInfo


ANLI_DATASET_PATH = "facebook/anli"
ANLI_DEFAULT_SPLIT = "dev_r3"
SUPPORTED_ANLI_SPLITS = (
    "train_r1",
    "dev_r1",
    "test_r1",
    "train_r2",
    "dev_r2",
    "test_r2",
    "train_r3",
    "dev_r3",
    "test_r3",
)
ANLI_BENCHMARK_INFO = BenchmarkInfo(
    name="ANLI",
    choices=("entailment", "neutral", "contradiction"),
)


@dataclass(frozen=True)
class ANLIExample:
    id: str
    premise: str
    hypothesis: str
    label: str
    reason: str

    @property
    def context(self) -> str:
        return f"Premise:\n{self.premise}"

    @property
    def question(self) -> str:
        return (
            "Classify the relationship between the hypothesis and the premise as one of the allowed choices.\n"
            f"Hypothesis:\n{self.hypothesis}"
        )

    @property
    def answers(self) -> List[str]:
        return [self.label]


def _normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in SUPPORTED_ANLI_SPLITS:
        raise ValueError(f"Unsupported ANLI split={split!r}; expected one of {SUPPORTED_ANLI_SPLITS}")
    return normalized


def _normalize_label(raw_label) -> str:
    choices = ANLI_BENCHMARK_INFO.choices
    if isinstance(raw_label, int):
        if not 0 <= raw_label < len(choices):
            raise ValueError(f"Unsupported ANLI label index: {raw_label}")
        return choices[raw_label]

    normalized = str(raw_label).strip().lower()
    for choice in choices:
        if normalized == choice.casefold():
            return choice
    raise ValueError(f"Unsupported ANLI label: {raw_label!r}")


def load_anli_examples(
    *,
    split: str = ANLI_DEFAULT_SPLIT,
    max_examples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[ANLIExample]:
    split = _normalize_split(split)
    dataset = load_dataset(ANLI_DATASET_PATH, split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=int(seed))

    rows = dataset if max_examples is None else itertools.islice(dataset, max(0, int(max_examples)))
    return [
        ANLIExample(
            id=str(row["uid"]),
            premise=str(row["premise"]).strip(),
            hypothesis=str(row["hypothesis"]).strip(),
            label=_normalize_label(row["label"]),
            reason=str(row.get("reason", "")).strip(),
        )
        for row in rows
    ]


__all__ = [
    "ANLI_DATASET_PATH",
    "ANLI_DEFAULT_SPLIT",
    "SUPPORTED_ANLI_SPLITS",
    "ANLI_BENCHMARK_INFO",
    "ANLIExample",
    "load_anli_examples",
]
