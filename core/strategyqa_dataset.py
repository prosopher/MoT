from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import List, Sequence
from urllib.error import URLError
from urllib.request import urlopen


STRATEGYQA_DEFAULT_DATA_DIR = "./strategyqa"
STRATEGYQA_DATA_URL = (
    "https://raw.githubusercontent.com/google/BIG-bench/main/"
    "bigbench/benchmark_tasks/strategyqa/task.json"
)
STRATEGYQA_TASK_FILENAME = "task.json"


@dataclass(frozen=True)
class StrategyQAExample:
    id: str
    question: str
    input_text: str
    reference: str
    choices: Sequence[str]

    @property
    def answers(self) -> List[str]:
        return [self.reference]


def _clean_text(text: str) -> str:
    # Match MALLM's DatasetDownloader._clean_text.
    return (
        str(text)
        .replace("\n", " ")
        .replace("\r", " ")
        .replace('"', "")
        .replace("\\n", " ")
    )


def ensure_strategyqa_file(*, data_dir: str = STRATEGYQA_DEFAULT_DATA_DIR) -> Path:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / STRATEGYQA_TASK_FILENAME
    if path.exists():
        return path

    try:
        with urlopen(STRATEGYQA_DATA_URL) as response, path.open("wb") as output:
            output.write(response.read())
    except (OSError, URLError) as exc:
        raise RuntimeError(
            f"Could not download StrategyQA from {STRATEGYQA_DATA_URL}. "
            f"Place the BIG-bench StrategyQA task.json at {path}."
        ) from exc
    return path


def load_strategyqa_examples(
    *,
    data_dir: str = STRATEGYQA_DEFAULT_DATA_DIR,
    max_examples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[StrategyQAExample]:
    path = ensure_strategyqa_file(data_dir=data_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("examples", []))

    indexed_rows = list(enumerate(rows, start=1))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indexed_rows)
    if max_examples is not None:
        indexed_rows = indexed_rows[:max_examples]

    examples: List[StrategyQAExample] = []
    for source_index, sample in indexed_rows:
        target_scores = sample.get("target_scores")
        if not isinstance(target_scores, dict) or not target_scores:
            raise ValueError(f"StrategyQA example {source_index} has no target_scores mapping")

        choice_texts = [_clean_text(str(choice)).strip() for choice in target_scores.keys()]
        normalized_choices = [choice.capitalize() for choice in choice_texts]
        if set(normalized_choices) != {"Yes", "No"}:
            raise ValueError(
                f"StrategyQA example {source_index} must use Yes/No target choices; got {choice_texts!r}"
            )
        correct = [
            normalized_choices[i]
            for i, score in enumerate(target_scores.values())
            if score == 1
        ]
        if len(correct) != 1:
            raise ValueError(
                f"StrategyQA example {source_index} must have exactly one correct answer; got {correct!r}"
            )

        question = _clean_text(str(sample.get("input", ""))).strip()
        if not question:
            raise ValueError(f"StrategyQA example {source_index} has an empty input")
        examples.append(
            StrategyQAExample(
                id=f"strategyqa-{source_index}",
                question=question,
                input_text=question,
                reference=correct[0],
                choices=tuple(normalized_choices),
            )
        )
    return examples


__all__ = [
    "STRATEGYQA_DEFAULT_DATA_DIR",
    "STRATEGYQA_DATA_URL",
    "STRATEGYQA_TASK_FILENAME",
    "StrategyQAExample",
    "ensure_strategyqa_file",
    "load_strategyqa_examples",
]
