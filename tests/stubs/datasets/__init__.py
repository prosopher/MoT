import random
from typing import Dict, Iterable, Iterator


class _BaseDataset:
    def __init__(self, items: Iterable[Dict]) -> None:
        self._items = list(items)

    def shuffle(self, seed: int = 42, buffer_size: int | None = None):
        rng = random.Random(seed)
        items = list(self._items)
        rng.shuffle(items)
        return self.__class__(items)

    def __iter__(self) -> Iterator[Dict]:
        return iter(self._items)


class FakeStreamingDataset(_BaseDataset):
    pass


class FakeMapDataset(_BaseDataset):
    pass


def load_dataset(dataset_path: str, dataset_name: str | None = None, split: str | None = None, streaming: bool = False):
    key = (dataset_path, dataset_name, split)

    if key == ("openwebtext", None, "train"):
        items = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"text": "Smoke tests should be fast, deterministic, and CPU-friendly."},
            {"text": "Tiny synthetic corpora are enough to exercise the training loop."},
        ]
        return FakeStreamingDataset(items)

    if key == ("google/boolq", None, "validation"):
        items = [
            {"question": "Is water wet?", "passage": "Water makes things wet.", "answer": True},
            {"question": "Is fire cold?", "passage": "Fire is hot.", "answer": False},
        ]
        return FakeMapDataset(items)

    if key == ("qiaojin/PubMedQA", "pqa_labeled", "train"):
        items = [
            {"question": "Does rest help recovery?", "context": {"contexts": ["Rest supports recovery."]}, "final_decision": "yes"},
            {"question": "Can rocks breathe?", "context": {"contexts": ["Rocks are not alive."]}, "final_decision": "no"},
        ]
        return FakeMapDataset(items)

    if key == ("rajpurkar/squad", None, "validation"):
        long_prefix = " ".join(["context"] * 80)
        items = [
            {
                "question": "What color is the sky?",
                "context": f"{long_prefix} On a clear day, the sky looks blue.",
                "answers": {"text": ["blue"]},
            },
            {
                "question": "What do bees make?",
                "context": "Bees are known for making honey.",
                "answers": {"text": ["honey"]},
            },
        ]
        return FakeMapDataset(items)


    if dataset_path == "edinburgh-dawg/mmlu-redux-2.0" and isinstance(dataset_name, str) and split == "test":
        items = [
            {
                "question": f"[{dataset_name}] What is 2 + 2?",
                "choices": ["3", "4", "5", "6"],
                "answer": 1,
                "error_type": "ok",
                "correct_answer": None,
            },
            {
                "question": f"[{dataset_name}] Which option is correct after relabeling?",
                "choices": ["wrong", "still wrong", "right answer", "also wrong"],
                "answer": 0,
                "error_type": "wrong_groundtruth",
                "correct_answer": "right answer",
            },
        ]
        return FakeMapDataset(items)

    if key == ("gabrieltorresgamez/newsqa", None, "validation"):
        long_prefix = " ".join(["news"] * 90)
        items = [
            {
                "paragraph": f"{long_prefix} The answer hidden in the report is blue.",
                "questions": ["What color is mentioned in the report?", "Which animal is discussed?"],
                "answers": [
                    {"text": ["blue"]},
                    {"text": ["fox"]},
                ],
            },
            {
                "paragraph": "The short article says the baker sold bread.",
                "questions": ["What did the baker sell?"],
                "answers": [{"text": ["bread"]}],
            },
        ]
        return FakeMapDataset(items)

    if key == ("Awesome075/multi_news_parquet", None, "validation"):
        long_prefix = " ||||| ".join([
            " ".join(["multinews"] * 45),
            "A coalition of editors produced a concise validation summary.",
        ])
        items = [
            {
                "document": long_prefix,
                "summary": "Editors produced a concise validation summary.",
            },
            {
                "document": "Report one describes the launch. ||||| Report two describes the reaction.",
                "summary": "The reports cover a launch and the reaction.",
            },
        ]
        return FakeMapDataset(items)

    raise ValueError(
        f"Unsupported fake dataset request: dataset_path={dataset_path!r}, dataset_name={dataset_name!r}, split={split!r}, streaming={streaming!r}"
    )
