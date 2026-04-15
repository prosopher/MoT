from types import SimpleNamespace

import torch

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



def build_profiler(layer_alignment: str) -> tuple[DeterministicChannelProfiler, Edge]:
    edge = Edge(id="A_to_B", src_id="A", tgt_id="B")
    ctx = Context(
        config=SimpleNamespace(alg="mot", layer_alignment=layer_alignment),
        nodes=[],
        edges=[edge],
        mm=DummyModelManager({"A": 2, "B": 4}),
        tokenizer=SimpleNamespace(),
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



def test_depth_ratio_selects_different_final_channels_than_terminal() -> None:
    terminal_profiler, edge = build_profiler("terminal")
    depth_ratio_profiler, _ = build_profiler("depth-ratio")

    terminal_result = terminal_profiler.profile_edge(edge)
    depth_ratio_result = depth_ratio_profiler.profile_edge(edge)

    assert [(channel.src_layer_idx, channel.dst_layer_idx) for channel in terminal_result.selected_channels] == [
        (0, 2),
        (1, 3),
    ]
    assert [(channel.src_layer_idx, channel.dst_layer_idx) for channel in depth_ratio_result.selected_channels] == [
        (0, 0),
        (1, 3),
    ]



def test_replay_interleaves_native_layers_between_translated_target_layers(monkeypatch) -> None:
    call_order: list[tuple[str, int]] = []

    transformer = SimpleNamespace(h=[SimpleNamespace(layer_idx=idx) for idx in range(12)])
    target_model = SimpleNamespace(transformer=transformer)

    def fake_build_gpt2_input_hidden_states(model, input_ids):
        del model, input_ids
        return torch.zeros(1, 1, 8)

    def fake_run_gpt2_block_with_cache(block, hidden_states):
        del hidden_states
        call_order.append(("cache", block.layer_idx))
        present = (
            torch.full((1, 2, 1, 4), float(block.layer_idx)),
            torch.full((1, 2, 1, 4), float(block.layer_idx)),
        )
        return torch.zeros(1, 1, 8), present

    def fake_run_gpt2_block_with_injected_layer(block, hidden_states, injected_key, injected_value):
        del hidden_states, injected_key, injected_value
        call_order.append(("inject", block.layer_idx))
        present = (
            torch.full((1, 2, 1, 4), float(block.layer_idx)),
            torch.full((1, 2, 1, 4), float(block.layer_idx)),
        )
        return torch.zeros(1, 1, 8), present

    monkeypatch.setattr(mot_train_module, "build_gpt2_input_hidden_states", fake_build_gpt2_input_hidden_states)
    monkeypatch.setattr(mot_train_module, "run_gpt2_block_with_cache", fake_run_gpt2_block_with_cache)
    monkeypatch.setattr(mot_train_module, "run_gpt2_block_with_injected_layer", fake_run_gpt2_block_with_injected_layer)

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
