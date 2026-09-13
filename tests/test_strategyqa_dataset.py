from __future__ import annotations

import json

from core.strategyqa_dataset import load_strategyqa_examples


def test_load_strategyqa_examples_from_local_json(tmp_path) -> None:
    rows = [
        {"qid": "q1", "question": "Question one?", "answer": True, "facts": ["fact 1"]},
        {"qid": "q2", "question": "Question two?", "answer": False, "facts": ["fact 2"]},
    ]
    (tmp_path / "dev.json").write_text(json.dumps(rows), encoding="utf-8")

    examples = load_strategyqa_examples(data_dir=str(tmp_path), split="dev")

    assert [example.id for example in examples] == ["q1", "q2"]
    assert [example.answers[0] for example in examples] == ["yes", "no"]
