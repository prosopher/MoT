from core.eval_util import (
    MMLU_REDUX_SUBJECTS,
    extract_question_and_answer,
    get_mmlu_redux_dataset_spec,
    is_logit_answer_correct,
)


def test_mmlu_redux_spec_uses_all_subject_configs() -> None:
    spec = get_mmlu_redux_dataset_spec()
    assert spec.dataset_path == "edinburgh-dawg/mmlu-redux-2.0"
    assert spec.dataset_names == MMLU_REDUX_SUBJECTS
    assert len(spec.dataset_names) == 57
    assert spec.answer_mode == "mmlu_redux"



def test_extract_mmlu_redux_ok_and_corrected_answers() -> None:
    spec = get_mmlu_redux_dataset_spec()

    ok_example = {
        "question": "What is 2 + 2?",
        "choices": ["3", "4", "5", "6"],
        "answer": 1,
        "error_type": "ok",
        "correct_answer": None,
        "subject": "elementary_mathematics",
    }
    wrong_groundtruth_example = {
        "question": "Which choice was relabeled?",
        "choices": ["old", "wrong", "new correct", "other"],
        "answer": 0,
        "error_type": "wrong_groundtruth",
        "correct_answer": "new correct",
        "subject": "miscellaneous",
    }
    multi_answer_example = {
        "question": "Which answers are both acceptable?",
        "choices": ["alpha", "beta", "gamma", "delta"],
        "answer": 0,
        "error_type": "multiple_correct_answers",
        "correct_answer": "B",
        "subject": "miscellaneous",
    }

    ok_pair = extract_question_and_answer(spec, ok_example)
    corrected_pair = extract_question_and_answer(spec, wrong_groundtruth_example)
    multi_pair = extract_question_and_answer(spec, multi_answer_example)

    assert ok_pair is not None
    assert ok_pair["answer"] == "B"
    assert ok_pair["choices"] == ["3", "4", "5", "6"]
    assert ok_pair["subject"] == "elementary_mathematics"

    assert corrected_pair is not None
    assert corrected_pair["answer"] == "C"

    assert multi_pair is not None
    assert multi_pair["answer"] == ["A", "B"]
    assert is_logit_answer_correct("A", multi_pair["answer"])
    assert is_logit_answer_correct("B", multi_pair["answer"])
    assert not is_logit_answer_correct("C", multi_pair["answer"])



def test_extract_mmlu_redux_skips_unscorable_examples() -> None:
    spec = get_mmlu_redux_dataset_spec()

    expert_example = {
        "question": "Need expert review?",
        "choices": ["A1", "B1", "C1", "D1"],
        "answer": 0,
        "error_type": "expert",
        "correct_answer": None,
        "subject": "miscellaneous",
    }
    no_correct_answer_example = {
        "question": "None of the options match.",
        "choices": ["A1", "B1", "C1", "D1"],
        "answer": 0,
        "error_type": "no_correct_answer",
        "correct_answer": "completely different answer",
        "subject": "miscellaneous",
    }

    assert extract_question_and_answer(spec, expert_example) is None
    assert extract_question_and_answer(spec, no_correct_answer_example) is None
