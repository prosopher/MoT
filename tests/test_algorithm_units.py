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
    align_sparse_attention_indices_to_target_tokens,
    align_cache_blocks_to_target_tokens,
    build_cross_token_alignment,
    collect_mot_balance_metrics,
    extract_source_attention_topk_indices,
    translate_layer_window,
)
from core.common import TokenIDs
from core.model_spec import ModelSpec
from core.channel_manager import Channel
from core.topology import Edge, Node, get_translator_id


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


class _FakeAttentionLayer(torch.nn.Module):
    def __init__(self, layer_idx: int, num_heads: int = 3) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads

    def forward(self, hidden_states, *, output_attentions=False, **_):
        # Deterministic per-layer probabilities. Hidden-state evolution is
        # independent of whether the caller retains the returned attention.
        batch_size, seq_len = hidden_states.shape[:2]
        base = torch.arange(
            batch_size * self.num_heads * seq_len * seq_len,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).reshape(batch_size, self.num_heads, seq_len, seq_len)
        attn = torch.softmax(base / float(11 + self.layer_idx), dim=-1)
        next_hidden = hidden_states + float(self.layer_idx + 1)
        return (next_hidden, attn) if output_attentions else (next_hidden,)


class _FakeRotaryDecoder(torch.nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Identity()
        self.layers = torch.nn.ModuleList([_FakeAttentionLayer(idx) for idx in range(num_layers)])


class _FakeRotaryCausalLM(torch.nn.Module):
    def __init__(self, num_layers: int = 4) -> None:
        super().__init__()
        self.model = _FakeRotaryDecoder(num_layers)

    def forward(self, input_ids, output_attentions=False, return_dict=True, **_):
        hidden = input_ids.to(torch.float32).unsqueeze(-1)
        attentions = []
        for layer in self.model.layers:
            outputs = layer(hidden, output_attentions=output_attentions)
            hidden = outputs[0]
            if output_attentions:
                attentions.append(outputs[1])
        if not return_dict:
            return (hidden, tuple(attentions) if output_attentions else None)
        return SimpleNamespace(
            last_hidden_state=hidden,
            attentions=tuple(attentions) if output_attentions else None,
        )


def test_mot_attention_topk_release_keeps_exact_indices() -> None:
    model = _FakeRotaryCausalLM(num_layers=4).eval()
    token_tensor = torch.tensor([[1, 3, 5, 7, 9]], dtype=torch.long)
    token_ids = TokenIDs(token_tensor, model_id="tiny")
    requested_layers = [0, 2, 3]
    top_k = 3

    with torch.no_grad():
        baseline_outputs = model(
            input_ids=token_tensor,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    baseline = []
    for layer_idx in requested_layers:
        shared = baseline_outputs.attentions[layer_idx].detach().mean(dim=1, keepdim=True)
        baseline.append(torch.topk(shared, k=top_k, dim=-1).indices)

    optimized = extract_source_attention_topk_indices(
        model,
        token_ids,
        requested_layers,
        source_model_id="Qwen/Qwen2-tiny",
        top_k=top_k,
    )

    assert len(optimized) == len(baseline)
    assert all(torch.equal(expected, actual) for expected, actual in zip(baseline, optimized))

    # Hooks must be temporary; ordinary callers still receive full attentions.
    with torch.no_grad():
        after = model(
            input_ids=token_tensor,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    assert all(attn is not None for attn in after.attentions)


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


class SpanTokenizer:
    def __init__(self, ids, offsets, pieces):
        self.ids = list(ids)
        self.offsets = list(offsets)
        self.pieces = dict(pieces)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_):
        assert text == "abcd"
        result = {"input_ids": list(self.ids)}
        if return_offsets_mapping:
            result["offset_mapping"] = list(self.offsets)
        return result

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self.pieces[int(token_id)] for token_id in token_ids)


def test_mot_cross_token_alignment_uses_text_spans_not_sequence_interpolation() -> None:
    source_model = SimpleNamespace(
        id="source",
        tokenizer=SpanTokenizer(
            ids=[10, 11, 12],
            offsets=[(0, 1), (1, 3), (3, 4)],
            pieces={10: "a", 11: "bc", 12: "d"},
        ),
    )
    target_model = SimpleNamespace(
        id="target",
        tokenizer=SpanTokenizer(
            ids=[20, 21],
            offsets=[(0, 3), (3, 4)],
            pieces={20: "abc", 21: "d"},
        ),
    )
    source_ids = TokenIDs(torch.tensor([[10, 11, 12]]), model_id="source")
    target_ids = TokenIDs(torch.tensor([[20, 21]]), model_id="target")

    weights, target_to_source, source_to_target = build_cross_token_alignment(
        source_model=source_model,
        target_model=target_model,
        source_context_token_ids=source_ids,
        target_context_token_ids=target_ids,
    )

    assert weights.shape == (1, 2, 3)
    assert torch.allclose(weights[0, 0], torch.tensor([1.0 / 3.0, 2.0 / 3.0, 0.0]))
    assert torch.allclose(weights[0, 1], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(target_to_source, torch.tensor([[1, 2]]))
    assert torch.equal(source_to_target, torch.tensor([[0, 0, 1]]))

    source_cache = torch.tensor([[[[3.0]], [[6.0]], [[9.0]]]])
    aligned_key, aligned_value = align_cache_blocks_to_target_tokens(
        source_cache,
        source_cache,
        alignment_weights=weights,
    )
    assert aligned_key.shape == (1, 2, 1, 1)
    assert torch.allclose(aligned_key.flatten(), torch.tensor([5.0, 9.0]))
    assert torch.equal(aligned_key, aligned_value)

    sparse = torch.tensor([[[[0, 0], [0, 1], [1, 2]]]])
    aligned_sparse = align_sparse_attention_indices_to_target_tokens(
        sparse,
        target_to_source=target_to_source,
        source_to_target=source_to_target,
    )
    assert aligned_sparse.shape == (1, 1, 2, 2)
    assert torch.equal(aligned_sparse, torch.tensor([[[[0, 2], [0, 1]]]]))


def test_mot_aligns_source_cache_before_translator() -> None:
    class CaptureTranslator:
        def __init__(self):
            self.key_input = None

        def __call__(self, key_block, value_block):
            self.key_input = key_block.detach().clone()
            return key_block, value_block

    edge = Edge(id="A_to_B", src_id="A", tgt_id="B")
    nodes = [Node(id="A", model_id="src-model"), Node(id="B", model_id="tgt-model")]
    translator = CaptureTranslator()
    ctx = SimpleNamespace(
        edges=[edge],
        nodes=nodes,
        cm=SimpleNamespace(get_channels=lambda edge_id: [Channel(src_layer_idx=0, dst_layer_idx=0)]),
        tp=SimpleNamespace(
            translators={get_translator_id("src-model", "tgt-model"): translator},
        ),
    )
    source_key = torch.tensor([[[[3.0], [6.0], [9.0]]]])
    source_value = source_key + 10.0
    weights = torch.tensor([[[1.0 / 3.0, 2.0 / 3.0, 0.0], [0.0, 0.0, 1.0]]])

    translated_key, translated_value = translate_layer_window(
        ctx,
        past_key_values=((source_key, source_value),),
        src_node_id="A",
        tgt_node_id="B",
        token_alignment_weights=weights,
    )

    assert translator.key_input is not None
    assert translator.key_input.shape == (1, 2, 1, 1)
    assert torch.allclose(translator.key_input.flatten(), torch.tensor([5.0, 9.0]))
    assert torch.equal(translated_key, translator.key_input)
    assert translated_value.shape == translated_key.shape
