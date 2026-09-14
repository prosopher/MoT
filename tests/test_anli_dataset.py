from __future__ import annotations

from core.anli_dataset import ANLI_BENCHMARK_INFO, ANLI_DEFAULT_SPLIT, load_anli_examples


def test_load_anli_examples_maps_labels_and_builds_nli_prompt() -> None:
    examples = load_anli_examples(split="dev_r3")

    assert ANLI_DEFAULT_SPLIT == "dev_r3"
    assert ANLI_BENCHMARK_INFO.choices == ("entailment", "neutral", "contradiction")
    assert [example.id for example in examples] == ["anli-e", "anli-n", "anli-c"]
    assert [example.answers[0] for example in examples] == ["entailment", "neutral", "contradiction"]
    assert examples[0].context == "Premise:\nA dog is running through a park."
    assert "Hypothesis:\nAn animal is outdoors." in examples[0].question
