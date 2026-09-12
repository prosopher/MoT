from types import SimpleNamespace

import pytest
import torch

from alg.c2c.train import normalize_top_layers_to_translate, resolve_top_layers_to_translate
from alg.interlat.train import (
    adjust_interlat_loss_weights,
    build_mismatched_source_hidden_states,
    compute_plan_similarity_loss,
    compute_random_contrast_loss,
)
from alg.kvcomm.train import KVCommSelectionTranslator, _resolve_candidate_target_layers, target_to_source_layer_map
from alg.lsc.train import blocks_to_past_key_values
from alg.mot.train import (
    MixtureOfTranslators,
    collect_mot_balance_metrics,
    resize_sparse_attention_indices,
    resize_sequence_cache,
)
from core.common import TokenIDs
from core.model_spec import ModelSpec


class ConstantTranslator(torch.nn.Module):
    next_value = 0

    def __init__(
        self,
        *,
        src_hidden_size: int,
        tgt_hidden_size: int,
        num_layers: int,
        translator_dim: int,
        translator_heads: int,
        translator_depth: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        del src_hidden_size, translator_dim, translator_heads, translator_depth, mlp_ratio
        self.value = float(ConstantTranslator.next_value)
        ConstantTranslator.next_value += 1
        self.num_layers = num_layers
        self.tgt_hidden_size = tgt_hidden_size

    def forward(self, layer_window_cache: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = layer_window_cache.shape[:2]
        return torch.full(
            (batch_size, seq_len, self.num_layers, self.tgt_hidden_size),
            self.value,
            dtype=layer_window_cache.dtype,
            device=layer_window_cache.device,
        )


def test_c2c_terminal_alignment_alias_resolves_to_edge_specific_min_depth() -> None:
    src_spec = ModelSpec(model_id="src", num_layers=3, hidden_size=8, num_heads=2, head_dim=4)
    tgt_spec = ModelSpec(model_id="tgt", num_layers=5, hidden_size=8, num_heads=2, head_dim=4)

    assert normalize_top_layers_to_translate("full") == "terminal_alignment"
    assert resolve_top_layers_to_translate("terminal-alignment", src_spec, tgt_spec) == 3
    assert resolve_top_layers_to_translate("2", src_spec, tgt_spec) == 2
    with pytest.raises(ValueError, match="A_to_B"):
        resolve_top_layers_to_translate(4, src_spec, tgt_spec, edge_id="A_to_B")


def test_interlat_auxiliary_losses_ignore_masked_labels_and_dynamic_weights_are_bounded() -> None:
    normal_logits = torch.tensor([[[4.0, 0.0, -1.0], [0.0, 4.0, -1.0]]])
    plan_logits = normal_logits.clone()
    random_logits = normal_logits.clone()
    labels = TokenIDs(torch.tensor([[0, -100]]), model_id="tiny")

    plan_loss = compute_plan_similarity_loss(
        normal_logits=normal_logits,
        plan_logits=plan_logits,
        label_token_ids=labels,
    )
    random_loss = compute_random_contrast_loss(
        normal_logits=normal_logits,
        random_logits=random_logits,
        label_token_ids=labels,
    )
    plan_weight, random_weight = adjust_interlat_loss_weights(
        random_contrast_loss=random_loss,
        plan_similarity_loss=plan_loss,
        initial_plan_weight=9.0,
        initial_random_weight=9.0,
    )

    assert plan_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert random_loss.item() > 0.0
    assert 0.01 <= plan_weight <= 0.05
    assert 0.01 <= random_weight <= 0.50


def test_interlat_mismatched_source_hidden_states_rolls_batches_without_aliasing() -> None:
    source_hidden_states = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)

    mismatched = build_mismatched_source_hidden_states(source_hidden_states)

    assert torch.equal(mismatched[0], source_hidden_states[1])
    assert torch.equal(mismatched[1], source_hidden_states[0])
    assert mismatched.requires_grad is False


def test_kvcomm_manual_layer_selection_validates_range_and_maps_source_layers() -> None:
    config = SimpleNamespace(layers_list=[3, 1, 1], top_layers=0.5)

    candidate_layers, manual_layers, num_layers_to_select = _resolve_candidate_target_layers(
        config=config,
        target_num_layers=4,
    )

    assert candidate_layers == [1, 3]
    assert manual_layers == [3, 1, 1]
    assert num_layers_to_select == 2
    assert target_to_source_layer_map(num_source_layers=2, num_target_layers=4) == {0: 0, 1: 0, 2: 1, 3: 1}

    bad_config = SimpleNamespace(layers_list=[4], top_layers=0.5)
    with pytest.raises(ValueError, match="outside target layer range"):
        _resolve_candidate_target_layers(config=bad_config, target_num_layers=4)


def test_kvcomm_selection_translator_round_trips_metadata_through_state_dict() -> None:
    original = KVCommSelectionTranslator(
        selected_target_layers=[1, 3],
        selected_source_layers=[0, 1],
        layer_ranking=[3, 1, 2, 0],
        calibration_score=0.25,
        attention_importance=[0.1, 0.2, 0.3, 0.4],
    )
    restored = KVCommSelectionTranslator(selected_target_layers=[])

    restored.load_state_dict(original.state_dict())

    assert restored.selected_target_layers == [1, 3]
    assert restored.selected_source_layers == [0, 1]
    assert restored.layer_ranking == [3, 1, 2, 0]
    assert restored.calibration_score == pytest.approx(0.25)
    assert restored.attention_importance == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_lsc_blocks_to_past_key_values_restores_layer_head_sequence_layout() -> None:
    model_spec = ModelSpec(model_id="tiny", num_layers=2, hidden_size=8, num_heads=2, head_dim=4)
    key_block = torch.arange(1 * 3 * 2 * 8, dtype=torch.float32).view(1, 3, 2, 8)
    value_block = key_block + 1000

    past_key_values = blocks_to_past_key_values(key_block, value_block, model_spec)

    assert len(past_key_values) == 2
    assert past_key_values[0][0].shape == (1, 2, 3, 4)
    assert torch.equal(past_key_values[1][0][:, :, 2, :].reshape(-1), key_block[:, 2, 1, :].reshape(-1))
    with pytest.raises(ValueError, match="Hidden mismatch"):
        blocks_to_past_key_values(key_block[..., :4], value_block[..., :4], model_spec)


def test_mot_top_k_router_masks_inactive_experts_and_exposes_balance_metrics() -> None:
    ConstantTranslator.next_value = 0
    module = MixtureOfTranslators(
        src_hidden_size=4,
        tgt_hidden_size=2,
        num_layers=2,
        translator_dim=4,
        translator_heads=1,
        translator_depth=1,
        mlp_ratio=1,
        num_translators=3,
        top_k=1,
        translator_cls=ConstantTranslator,
    )
    module.eval()
    for param in module.router.parameters():
        param.data.zero_()
    module.router[-1].bias.data.copy_(torch.tensor([0.0, 10.0, 1.0]))

    output = module(torch.randn(2, 3, 2, 4))

    assert torch.allclose(output, torch.ones_like(output))
    assert torch.isneginf(module.last_router_logits[..., 0]).all()
    assert torch.isneginf(module.last_router_logits[..., 2]).all()
    assert module.last_mixture_weights[..., 1].eq(1.0).all()
    assert set(collect_mot_balance_metrics(module)) == {
        "gate_importance_cv2",
        "gate_load_cv2",
        "gate_importance_entropy",
    }


def test_mot_sparse_indices_resize_between_tokenizer_lengths() -> None:
    source_indices = torch.tensor([[[[0, 286], [1, 285], [143, 200]]]])

    resized = resize_sparse_attention_indices(
        source_indices,
        target_query_len=286,
        target_key_len=286,
    )

    assert resized.shape == (1, 1, 286, 2)
    assert resized.dtype == torch.long
    assert int(resized.min()) >= 0
    assert int(resized.max()) < 286
    assert torch.equal(resized[:, :, 0, :], source_indices[:, :, 0, :].clamp(max=285))

    expanded = source_indices.expand(-1, 4, -1, -1)
    expanded_resized = resize_sparse_attention_indices(
        expanded,
        target_query_len=286,
        target_key_len=286,
    )
    assert expanded_resized.shape == (1, 4, 286, 2)

    cache = torch.arange(1 * 1 * 287 * 2, dtype=torch.float32).view(1, 1, 287, 2)
    resized_cache = resize_sequence_cache(cache, target_seq_len=286)
    assert resized_cache.shape == (1, 1, 286, 2)
    assert torch.equal(resized_cache[:, :, 0, :], cache[:, :, 0, :])
    assert torch.equal(resized_cache[:, :, -1, :], cache[:, :, -1, :])
