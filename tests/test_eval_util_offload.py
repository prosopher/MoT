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


def test_gpu_memory_breakdown_uses_explicit_kv_storage_not_allocator_residual(monkeypatch) -> None:
    from core import eval_util

    monkeypatch.setattr(eval_util.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(eval_util.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(eval_util.torch.cuda, "synchronize", lambda _device_index: None)
    # Deliberately huge allocator usage: this must not leak into kv_bytes.
    monkeypatch.setattr(eval_util.torch.cuda, "memory_allocated", lambda _device_index: 10_000)

    module_calls = iter([100, 20])
    monkeypatch.setattr(eval_util, "_cuda_storage_bytes", lambda *args, **kwargs: next(module_calls))
    monkeypatch.setattr(eval_util, "_cuda_nested_storage_bytes", lambda *args, **kwargs: 30)

    measured = eval_util.measure_gpu_memory_breakdown_bytes(
        "cuda:0",
        models=[object()],
        translator_pool=object(),
        kv_objects=[object()],
    )

    assert measured is not None
    assert measured.model_bytes == 100
    assert measured.translator_bytes == 20
    assert measured.kv_bytes == 30
    assert measured.total_bytes == 150


def test_inference_profile_reports_only_model_translator_and_kv_memory() -> None:
    from core.eval_util import InferenceProfileAccumulator

    gib = 1024 ** 3
    accumulator = InferenceProfileAccumulator()
    accumulator.update(
        latency_sec=1.0,
        tokens=100,
        model_memory_bytes=10 * gib,
        translator_memory_bytes=2 * gib,
        kv_memory_bytes=3 * gib,
    )

    summary = accumulator.summary()
    assert summary["model_memory_gib"] == 10.0
    assert summary["translator_memory_gib"] == 2.0
    assert summary["kv_memory_gib"] == 3.0
    memory_keys = {key for key in summary if key.endswith("_memory_gib")}
    assert memory_keys == {"model_memory_gib", "translator_memory_gib", "kv_memory_gib"}
