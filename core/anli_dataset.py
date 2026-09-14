from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
from typing import List, Optional
from urllib.error import URLError
from urllib.request import urlopen
import zipfile


ANLI_DEFAULT_DATA_DIR = "./anli"
ANLI_DEFAULT_SPLIT = "dev_r3"
ANLI_ARCHIVE_URL = "https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip"
SUPPORTED_ANLI_SPLITS = ("train_r3", "dev_r3", "test_r3")
ANLI_LABELS = ("entailment", "neutral", "contradiction")


@dataclass(frozen=True)
class ANLIExample:
    id: str
    premise: str
    hypothesis: str
    label: str
    reason: str

    @property
    def answers(self) -> List[str]:
        return [self.label]


def _normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in SUPPORTED_ANLI_SPLITS:
        raise ValueError(f"Unsupported ANLI split={split!r}; expected one of {SUPPORTED_ANLI_SPLITS}")
    return normalized


def _split_filename(split: str) -> str:
    return _normalize_split(split).split("_", 1)[0]


def _anli_path(data_dir: str, split: str) -> Path:
    return Path(data_dir) / "anli_v1.0" / "R3" / f"{_split_filename(split)}.jsonl"


def ensure_anli_file(
    *,
    data_dir: str = ANLI_DEFAULT_DATA_DIR,
    split: str = ANLI_DEFAULT_SPLIT,
) -> Path:
    path = _anli_path(data_dir, split)
    if path.exists():
        return path

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "anli_v1.0.zip"
    if not archive_path.exists():
        try:
            with urlopen(ANLI_ARCHIVE_URL) as response, archive_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        except (OSError, URLError) as exc:
            raise RuntimeError(
                f"Could not download ANLI from {ANLI_ARCHIVE_URL}. "
                f"Download the archive manually and save it as {archive_path}."
            ) from exc

    member = f"anli_v1.0/R3/{_split_filename(split)}.jsonl"
    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extract(member, path=root)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"ANLI archive does not contain {member}: {archive_path}") from exc
    return path


def _normalize_label(label) -> str:
    aliases = {
        "0": "entailment",
        "1": "neutral",
        "2": "contradiction",
        "e": "entailment",
        "n": "neutral",
        "c": "contradiction",
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
    }
    normalized = str(label).strip().lower()
    if normalized not in aliases:
        raise ValueError(f"Unsupported ANLI label={label!r}")
    return aliases[normalized]


def load_anli_examples(
    *,
    data_dir: str = ANLI_DEFAULT_DATA_DIR,
    split: str = ANLI_DEFAULT_SPLIT,
    max_examples: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[ANLIExample]:
    path = ensure_anli_file(data_dir=data_dir, split=split)
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    if shuffle:
        random.Random(int(seed)).shuffle(rows)
    if max_examples is not None:
        rows = rows[: max(0, int(max_examples))]

    return [
        ANLIExample(
            id=str(row["uid"]),
            premise=str(row["context"] if "context" in row else row["premise"]).strip(),
            hypothesis=str(row["hypothesis"]).strip(),
            label=_normalize_label(row["label"]),
            reason=str(row.get("reason", "")).strip(),
        )
        for row in rows
    ]


__all__ = [
    "ANLI_DEFAULT_DATA_DIR",
    "ANLI_DEFAULT_SPLIT",
    "ANLI_ARCHIVE_URL",
    "ANLI_LABELS",
    "ANLIExample",
    "ensure_anli_file",
    "load_anli_examples",
]
