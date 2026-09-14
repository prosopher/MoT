from __future__ import annotations

import json

from core.anli_dataset import load_anli_examples


def test_load_anli_r3_examples_from_local_jsonl(tmp_path) -> None:
    data_dir = tmp_path
    r3_dir = data_dir / "anli_v1.0" / "R3"
    r3_dir.mkdir(parents=True)
    rows = [
        {"uid": "q1", "premise": "A cat sleeps.", "hypothesis": "An animal sleeps.", "label": "e", "reason": ""},
        {"uid": "q2", "premise": "It is raining.", "hypothesis": "It is sunny.", "label": "c", "reason": ""},
        {"uid": "q3", "premise": "Sam owns a car.", "hypothesis": "The car is red.", "label": "n", "reason": ""},
    ]
    (r3_dir / "dev.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    examples = load_anli_examples(data_dir=str(data_dir), split="dev_r3")

    assert [example.id for example in examples] == ["q1", "q2", "q3"]
    assert [example.label for example in examples] == ["entailment", "contradiction", "neutral"]
    assert examples[0].premise == "A cat sleeps."
    assert examples[0].hypothesis == "An animal sleeps."


def test_load_anli_r3_examples_accepts_context_field(tmp_path) -> None:
    r3_dir = tmp_path / "anli_v1.0" / "R3"
    r3_dir.mkdir(parents=True)
    row = {
        "uid": "q1",
        "context": "A cat sleeps.",
        "hypothesis": "An animal sleeps.",
        "label": "e",
        "reason": "",
    }
    (r3_dir / "dev.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    examples = load_anli_examples(data_dir=str(tmp_path), split="dev_r3")

    assert len(examples) == 1
    assert examples[0].premise == "A cat sleeps."
    assert examples[0].label == "entailment"
