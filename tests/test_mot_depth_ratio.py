import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from core.channel_manager import ChannelManager
from core.channel_profiler import ChannelProfileConfig, ChannelProfiler, ProxyValidationScore
from core.context import Context
from core.model_spec import ModelSpec
from core.topology import Edge
from mot.train import replay_target_prefill_with_injected_window
import mot.train as mot_train_module


class DummyModelManager:
    def __init__(self, layer_counts: dict[str, int]) -> None:
        self.layer_counts = layer_counts

    def get_model_spec(self, model_id: str):
        return SimpleNamespace(num_layers=self.layer_counts[model_id])


class DeterministicChannelProfiler(ChannelProfiler):
    def _build_profile_bank(self, edge: Edge, *, num_examples: int, split: str, seed_offset: int):
        del edge, num_examples, split, seed_offset
        return [{}]

    def _score_channels(self, edge: Edge, channels, train_bank, val_bank) -> ProxyValidationScore:
        del edge, channels, train_bank, val_bank
        return ProxyValidationScore(translated_loss=0.0, native_loss=0.0)


class ScoreMappedChannelProfiler(ChannelProfiler):
    def __init__(self, ctx: Context, profile_config: ChannelProfileConfig, score_map: dict[tuple[int, ...], float]) -> None:
        super().__init__(ctx, profile_config)
        self.score_map = score_map
        self.scored_windows: list[tuple[int, ...]] = []

    def _build_profile_bank(self, edge: Edge, *, num_examples: int, split: str, seed_offset: int):
        del edge, num_examples, split, seed_offset
        return [{}]

    def _score_channels(self, edge: Edge, channels, train_bank, val_bank) -> ProxyValidationScore:
        del edge, train_bank, val_bank
        key = tuple(channel.src_layer_idx for channel in channels)
        self.scored_windows.append(key)
        if key not in self.score_map:
            raise AssertionError(f"unexpected score request for window {key}")
        return ProxyValidationScore(translated_loss=self.score_map[key], native_loss=0.0)



def build_profiler(layer_alignment: str) -> tuple[DeterministicChannelProfiler, Edge]:
    edge = Edge(id="A_to_B", src_id="A", tgt_id="B")
    ctx = Context(
        config=SimpleNamespace(
            alg="mot",
            layer_alignment=layer_alignment,
            min_window_size_ratio=0.33,
            max_window_size_ratio=0.5,
        ),
        nodes=[],
        edges=[edge],
        mm=DummyModelManager({"A": 2, "B": 4}),
        cm=ChannelManager([edge]),
    )
    profiler = DeterministicChannelProfiler(
        ctx,
        ChannelProfileConfig(
            num_train_examples=1,
            num_val_examples=1,
            max_steps=0,
            min_window_size=1,
            translator_dim=8,
            translator_heads=1,
            translator_depth=1,
            translator_mlp_ratio=1,
        ),
    )
    return profiler, edge



def build_scored_profiler(
    layer_alignment: str,
    *,
    layer_counts: tuple[int, int],
    score_map: dict[tuple[int, ...], float],
) -> tuple[ScoreMappedChannelProfiler, Edge]:
    edge = Edge(id="A_to_B", src_id="A", tgt_id="B")
    ctx = Context(
        config=SimpleNamespace(
            alg="mot",
            layer_alignment=layer_alignment,
            min_window_size_ratio=0.33,
            max_window_size_ratio=0.5,
        ),
        nodes=[],
        edges=[edge],
        mm=DummyModelManager({"A": layer_counts[0], "B": layer_counts[1]}),
        cm=ChannelManager([edge]),
    )
    profiler = ScoreMappedChannelProfiler(
        ctx,
        ChannelProfileConfig(
            num_train_examples=1,
            num_val_examples=1,
            max_steps=0,
            min_window_size=1,
            translator_dim=8,
            translator_heads=1,
            translator_depth=1,
            translator_mlp_ratio=1,
        ),
        score_map=score_map,
    )
    return profiler, edge



def test_depth_ratio_selects_different_final_channels_than_terminal() -> None:
    terminal_profiler, edge = build_profiler("terminal")
    depth_ratio_profiler, _ = build_profiler("depth-ratio")

    terminal_result = terminal_profiler.profile_edge(edge)
    depth_ratio_result = depth_ratio_profiler.profile_edge(edge)

    assert [(channel.src_layer_idx, channel.dst_layer_idx) for channel in terminal_result.selected_channels] == [
        (0, 2),
    ]
    assert [(channel.src_layer_idx, channel.dst_layer_idx) for channel in depth_ratio_result.selected_channels] == [
        (0, 0),
    ]


@pytest.mark.parametrize("layer_alignment", ["terminal", "depth-ratio"])
def test_window_growth_reaches_max_probe_window_without_contain_requirement(
    layer_alignment: str,
) -> None:
    score_map = {
        (0, 1, 2, 3): 5.0,
        (1, 2, 3, 4): 4.0,
        (2, 3, 4, 5): 3.0,
        (3, 4, 5, 6): 2.0,
        (4, 5, 6, 7): 1.0,
        (5, 6, 7, 8): 0.2,
        (6, 7, 8, 9): 0.5,
        (7, 8, 9, 10): 0.8,
        (8, 9, 10, 11): 1.2,
        (3, 4, 5, 6, 7): 0.7,
        (4, 5, 6, 7, 8): 0.6,
        (5, 6, 7, 8, 9): 0.9,
        (6, 7, 8, 9, 10): 1.1,
        (7, 8, 9, 10, 11): 1.4,
        (2, 3, 4, 5, 6, 7): 0.1,
        (3, 4, 5, 6, 7, 8): 0.9,
        (4, 5, 6, 7, 8, 9): 1.3,
        (5, 6, 7, 8, 9, 10): 1.8,
        (6, 7, 8, 9, 10, 11): 2.0,
    }
    profiler, edge = build_scored_profiler(
        layer_alignment,
        layer_counts=(12, 12),
        score_map=score_map,
    )

    result = profiler.profile_edge(edge)

    assert [channel.src_layer_idx for channel in result.selected_channels] == [2, 3, 4, 5, 6, 7]
    assert result.best_validation_loss == 0.1
    assert [step.removed_side for step in result.history[-3:]] == ["expand-win-5", "expand-win-6", "selected"]


@pytest.mark.parametrize("layer_alignment", ["terminal", "depth-ratio"])
def test_probe_window_uses_one_third_start_half_max_and_keeps_edge_layers(layer_alignment: str) -> None:
    score_map: dict[tuple[int, ...], float] = {}
    for win_size in (4, 5, 6):
        for start in range(12 - win_size + 1):
            window = tuple(range(start, start + win_size))
            score_map[window] = float(start + win_size)
    profiler, edge = build_scored_profiler(
        layer_alignment,
        layer_counts=(12, 12),
        score_map=score_map,
    )

    result = profiler.profile_edge(edge)

    scored_len_4 = [window for window in profiler.scored_windows if len(window) == 4]
    assert tuple(range(0, 4)) in scored_len_4
    assert tuple(range(8, 12)) in scored_len_4
    assert not any(len(window) == 7 for window in profiler.scored_windows)
    assert len(result.selected_channels) == 6



def test_replay_interleaves_native_layers_between_translated_target_layers(monkeypatch) -> None:
    call_order: list[tuple[str, int]] = []

    transformer = SimpleNamespace(h=[SimpleNamespace(layer_idx=idx) for idx in range(12)])
    target_model = SimpleNamespace(transformer=transformer)

    def fake_build_gpt2_input_hidden_states(model, input_ids):
        del model, input_ids
        return torch.zeros(1, 1, 8)

    def fake_run_gpt2_block(block, hidden_states, *, sparse_attention_indices=None, injected_key=None, injected_value=None):
        del hidden_states, sparse_attention_indices
        call_order.append(("cache" if injected_key is None else "inject", block.layer_idx))
        present = (
            torch.full((1, 2, 1, 4), float(block.layer_idx)),
            torch.full((1, 2, 1, 4), float(block.layer_idx)),
        )
        return torch.zeros(1, 1, 8), present

    monkeypatch.setattr(mot_train_module, "build_gpt2_input_hidden_states", fake_build_gpt2_input_hidden_states)
    monkeypatch.setattr(mot_train_module, "run_gpt2_block", fake_run_gpt2_block)

    with torch.no_grad():
        replayed_past = replay_target_prefill_with_injected_window(
            target_model=target_model,
            prefix_input_ids=torch.tensor([[1]]),
            target_layer_indices=[1, 3, 5, 7, 9, 11],
            injected_key_block=torch.zeros(1, 1, 6, 8),
            injected_value_block=torch.zeros(1, 1, 6, 8),
            tgt_spec=ModelSpec(
                model_id="dummy-target",
                num_layers=12,
                hidden_size=8,
                num_heads=2,
                head_dim=4,
            ),
        )

    assert len(replayed_past) == 12
    assert call_order == [
        ("cache", 0),
        ("inject", 1),
        ("cache", 2),
        ("inject", 3),
        ("cache", 4),
        ("inject", 5),
        ("cache", 6),
        ("inject", 7),
        ("cache", 8),
        ("inject", 9),
        ("cache", 10),
        ("inject", 11),
    ]



def test_resolve_target_model_family_accepts_pythia_gpt_neox() -> None:
    assert mot_train_module.normalize_model_family("EleutherAI/pythia-70m") == "gpt_neox"
    assert mot_train_module.normalize_model_family("EleutherAI/gpt-neox-20b") == "gpt_neox"

    target_model = SimpleNamespace(
        config=SimpleNamespace(model_type="gpt_neox"),
        gpt_neox=SimpleNamespace(layers=[], embed_in=object()),
    )

    assert mot_train_module.resolve_target_model_family(target_model) == "gpt_neox"



def test_replay_dispatches_pythia_gpt_neox_layers(monkeypatch) -> None:
    call_order: list[tuple[str, int]] = []

    gpt_neox = SimpleNamespace(
        layers=[SimpleNamespace(layer_idx=idx) for idx in range(4)],
        embed_in=object(),
        training=False,
    )
    target_model = SimpleNamespace(config=SimpleNamespace(model_type="gpt_neox"), gpt_neox=gpt_neox)

    def fake_build_gpt_neox_input_hidden_states(model, input_ids):
        del model
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.zeros(batch_size, seq_len, 8)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        return hidden_states, position_ids, attention_mask, None

    def fake_run_gpt_neox_block(
        block,
        hidden_states,
        *,
        position_ids,
        attention_mask,
        sparse_attention_indices=None,
        injected_key=None,
        injected_value=None,
    ):
        del position_ids, attention_mask, sparse_attention_indices, injected_value
        call_order.append(("cache" if injected_key is None else "inject", block.layer_idx))
        present = (
            torch.full((1, 2, 1, 4), float(block.layer_idx)),
            torch.full((1, 2, 1, 4), float(block.layer_idx)),
        )
        return hidden_states, present

    monkeypatch.setattr(mot_train_module, "build_gpt_neox_input_hidden_states", fake_build_gpt_neox_input_hidden_states)
    monkeypatch.setattr(mot_train_module, "run_gpt_neox_block", fake_run_gpt_neox_block)

    with torch.no_grad():
        replayed_past = replay_target_prefill_with_injected_window(
            target_model=target_model,
            prefix_input_ids=torch.tensor([[1]]),
            target_layer_indices=[1, 3],
            injected_key_block=torch.zeros(1, 1, 2, 8),
            injected_value_block=torch.zeros(1, 1, 2, 8),
            tgt_spec=ModelSpec(
                model_id="pythia-target",
                num_layers=4,
                hidden_size=8,
                num_heads=2,
                head_dim=4,
            ),
        )

    assert len(replayed_past) == 4
    assert call_order == [
        ("cache", 0),
        ("inject", 1),
        ("cache", 2),
        ("inject", 3),
    ]


class TinyGPTNeoXAttention(nn.Module):
    def __init__(self, hidden_size: int = 8, num_heads: int = 2) -> None:
        super().__init__()
        self.num_attention_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.rotary_ndims = 0
        self.norm_factor = math.sqrt(self.head_size)
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = 0.0


class TinyGPTNeoXBlock(nn.Module):
    def __init__(self, hidden_size: int = 8, num_heads: int = 2, use_parallel_residual: bool = True) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.attention = TinyGPTNeoXAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(), nn.Linear(4 * hidden_size, hidden_size))
        self.post_attention_dropout = nn.Dropout(0.0)
        self.post_mlp_dropout = nn.Dropout(0.0)
        self.use_parallel_residual = use_parallel_residual


@pytest.mark.parametrize("use_parallel_residual", [True, False])
def test_run_gpt_neox_block_rebuilds_native_cache(use_parallel_residual: bool) -> None:
    torch.manual_seed(123)
    block = TinyGPTNeoXBlock(use_parallel_residual=use_parallel_residual)
    block.eval()
    hidden_states = torch.randn(1, 3, 8)
    position_ids = torch.arange(3).unsqueeze(0)
    attention_mask = mot_train_module.build_causal_attention_mask(hidden_states)

    output, present = mot_train_module.run_gpt_neox_block(
        block,
        hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )

    assert output.shape == hidden_states.shape
    assert present[0].shape == (1, 2, 3, 4)
    assert present[1].shape == (1, 2, 3, 4)
    assert torch.isfinite(output).all()
