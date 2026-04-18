import time

from core import eval_util


class SequenceReader:
    def __init__(self, values):
        self.values = list(values)
        self.index = 0
        self.enabled = True

    def read_allocated_bytes(self):
        if self.index >= len(self.values):
            return self.values[-1]
        value = self.values[self.index]
        self.index += 1
        return value


def _make_enabled_profiler(monkeypatch, *, max_peaks):
    profiler = eval_util.InferenceProfiler("cpu", sample_interval_sec=0.001)
    profiler.enabled = True
    profiler.device_index = 0

    peak_iter = iter(max_peaks)
    reset_calls = []
    sync_calls = []

    monkeypatch.setattr(eval_util.torch.cuda, "synchronize", lambda device=None: sync_calls.append(device))
    monkeypatch.setattr(
        eval_util.torch.cuda,
        "reset_peak_memory_stats",
        lambda device=None: reset_calls.append(device),
    )
    monkeypatch.setattr(
        eval_util.torch.cuda,
        "max_memory_allocated",
        lambda device=None: next(peak_iter),
    )
    return profiler, reset_calls, sync_calls


def test_inference_profiler_reports_absolute_peak_memory(monkeypatch):
    profiler, reset_calls, _ = _make_enabled_profiler(monkeypatch, max_peaks=[170])
    profiler.reader = SequenceReader([100, 140, 160, 130])

    result, profile = profiler.measure(lambda: (time.sleep(0.01), "ok")[1], tokens=5)

    assert result == "ok"
    assert profile["tokens"] == 5
    assert profile["peak_memory_bytes"] == 170
    assert reset_calls == [0]



def test_inference_profiler_resets_peak_stats_between_measurements(monkeypatch):
    profiler, reset_calls, _ = _make_enabled_profiler(monkeypatch, max_peaks=[155, 222])

    profiler.reader = SequenceReader([100, 150, 120])
    _, first_profile = profiler.measure(lambda: time.sleep(0.005), tokens=1)

    profiler.reader = SequenceReader([200, 220, 205])
    _, second_profile = profiler.measure(lambda: time.sleep(0.005), tokens=1)

    assert first_profile["peak_memory_bytes"] == 155
    assert second_profile["peak_memory_bytes"] == 222
    assert reset_calls == [0, 0]
