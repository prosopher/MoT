import pytest

from core.eval_util import temporarily_offload_module


class _MovablePool:
    """Minimal TranslatorPool-like object: not nn.Module, but exposes to()."""

    def __init__(self) -> None:
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self


def _mock_cuda(monkeypatch) -> list[int]:
    synchronized = []
    monkeypatch.setattr("core.eval_util.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("core.eval_util.torch.cuda.synchronize", lambda device_index: synchronized.append(device_index))
    return synchronized


def test_temporarily_offload_module_supports_translator_pool_like_object(monkeypatch) -> None:
    synchronized = _mock_cuda(monkeypatch)
    pool = _MovablePool()

    with temporarily_offload_module(pool, "cuda:2"):
        assert pool.moves == ["cpu"]

    assert pool.moves == ["cpu", "cuda:2"]
    assert synchronized == [2, 2]


def test_temporarily_offload_module_restores_translator_pool_like_object_on_error(monkeypatch) -> None:
    synchronized = _mock_cuda(monkeypatch)
    pool = _MovablePool()

    with pytest.raises(RuntimeError, match="boom"):
        with temporarily_offload_module(pool, "cuda:1"):
            raise RuntimeError("boom")

    assert pool.moves == ["cpu", "cuda:1"]
    assert synchronized == [1, 1]
