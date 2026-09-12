from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
STUBS_PATH = REPO_ROOT / "tests" / "stubs"
if str(STUBS_PATH) not in sys.path:
    sys.path.insert(0, str(STUBS_PATH))

from transformers import AutoConfig, TinyTokenizer


def test_tiny_tokenizer_supports_truncation_and_max_length() -> None:
    tokenizer = TinyTokenizer("tiny-a")
    encoded = tokenizer("abcdef", add_special_tokens=False, truncation=True, max_length=3)
    assert encoded.input_ids == tokenizer._encode_text("abc")


def test_tiny_tokenizer_exposes_small_model_max_length_for_smoke_eval() -> None:
    tokenizer = TinyTokenizer("tiny-a")
    assert tokenizer.model_max_length == 128


def test_stub_autoconfig_matches_tiny_model_shape_contract() -> None:
    config = AutoConfig.from_pretrained("tiny-a")
    assert config.n_layer == 2
    assert config.n_head == 2
    assert config.n_embd == 8


def test_eval_text_preparation_matches_training_special_token_policy() -> None:
    from types import SimpleNamespace
    from core.eval_util import prepare_cache_text_inputs, prepare_full_text_inputs

    tokenizer = TinyTokenizer("tiny-a")
    model = SimpleNamespace(id="tiny-a", tokenizer=tokenizer)

    full = prepare_full_text_inputs(model=model, text="abcd", device="cpu")
    cached = prepare_cache_text_inputs(model=model, text="abcd", device="cpu")

    expected = tokenizer("abcd", add_special_tokens=False).input_ids
    assert full["token_ids"].shape[1] == len(expected)
    assert cached["token_ids"].shape[1] == len(expected)
