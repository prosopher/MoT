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
import alg.lsc.eval as lsc_eval
from alg.mot.train import (
    MixtureOfTranslators,
    align_sparse_attention_indices_to_target_tokens,
    align_cache_blocks_to_target_tokens,
    build_cross_token_alignment,
    build_replayed_target_past,
    build_causal_attention_mask,
    collect_mot_balance_metrics,
    extract_source_attention_topk_indices,
    translate_layer_window,
    _build_rotary_input_hidden_states,
)
from core.common import TokenIDs, extract_receiver_aligned_sharer_past
from core.model import Model
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


class _AlignmentTokenizer:
    def __init__(self, *, decode_map, encode_map, special_ids=None) -> None:
        self.decode_map = {int(key): value for key, value in decode_map.items()}
        self.encode_map = {str(key): list(value) for key, value in encode_map.items()}
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.all_special_ids = list(special_ids or [0, 1, 2, 3])

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self.decode_map.get(int(token_id), f"<{int(token_id)}>") for token_id in token_ids)

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        del add_special_tokens, return_tensors
        return list(self.encode_map.get(text, [self.unk_token_id]))

    def convert_tokens_to_ids(self, token_text):
        candidates = self.encode_map.get(token_text)
        if not candidates:
            return self.unk_token_id
        return int(candidates[0])


class _PastOnlyLM(torch.nn.Module):
    def __init__(self, *, num_layers=2, num_heads=1, head_dim=2) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, input_ids, use_cache=True, **_):
        del use_cache
        batch_size, seq_len = input_ids.shape
        past = []
        values = input_ids.to(torch.float32).view(batch_size, 1, seq_len, 1)
        values = values.expand(batch_size, self.num_heads, seq_len, self.head_dim).contiguous()
        for layer_idx in range(self.num_layers):
            key = values + float(layer_idx)
            value = values + float(100 + layer_idx)
            past.append((key, value))
        return SimpleNamespace(past_key_values=tuple(past))


def _build_cross_tokenizer_test_models():
    receiver_tokenizer = _AlignmentTokenizer(
        decode_map={10: "ab", 11: "c", 12: "def", 0: "<pad>", 1: "<eos>", 2: "<bos>", 3: "<unk>"},
        encode_map={"ab": [10], "c": [11], "def": [12]},
    )
    sharer_tokenizer = _AlignmentTokenizer(
        decode_map={20: "a", 21: "ab", 22: "c", 23: "d", 24: "def", 0: "<pad>", 1: "<eos>", 2: "<bos>", 3: "<unk>"},
        encode_map={"ab": [20, 21], "c": [22], "def": [23, 24]},
    )
    receiver_model = Model("receiver", _PastOnlyLM(), receiver_tokenizer)
    sharer_model = Model("sharer", _PastOnlyLM(), sharer_tokenizer)
    return receiver_model, sharer_model


def test_cross_tokenizer_alignment_uses_receiver_length_and_c2c_longest_rule() -> None:
    receiver_model, sharer_model = _build_cross_tokenizer_test_models()
    receiver_tokens = TokenIDs(torch.tensor([[10, 11, 12]]), model_id=receiver_model.id)

    aligned_past = extract_receiver_aligned_sharer_past(
        receiver_context_token_ids=receiver_tokens,
        receiver_model=receiver_model,
        sharer_model=sharer_model,
    )

    # "ab" -> [20, 21] selects 21 because its decoded text covers more of the
    # receiver token; "def" analogously selects 24. Most importantly, the
    # resulting sharer KV has exactly the receiver's three token positions.
    expected_ids = torch.tensor([21.0, 22.0, 24.0])
    assert aligned_past[0][0].shape[2] == 3
    assert torch.equal(aligned_past[0][0][0, 0, :, 0], expected_ids)


def test_lsc_logit_eval_aligns_cross_tokenized_source_to_target_length(monkeypatch) -> None:
    receiver_model, sharer_model = _build_cross_tokenizer_test_models()
    receiver_tokens = TokenIDs(torch.tensor([[10, 11, 12]]), model_id=receiver_model.id)
    source_tokens = TokenIDs(torch.tensor([[20, 21, 22, 23, 24]]), model_id=sharer_model.id)

    with torch.no_grad():
        native_target_past = receiver_model(input_ids=receiver_tokens.as_tensor(), use_cache=True).past_key_values
        mismatched_source_past = sharer_model(input_ids=source_tokens.as_tensor(), use_cache=True).past_key_values

    class _TP:
        def get_model(self, node_id):
            return {"A": sharer_model, "B": receiver_model}[node_id]

        def get_model_spec(self, node_id):
            del node_id
            return ModelSpec(model_id="receiver", num_layers=2, hidden_size=2, num_heads=1, head_dim=2)

    ctx = SimpleNamespace(tp=_TP())
    edge = Edge(id="A_to_B", src_id="A", tgt_id="B")
    seen = {}

    def fake_translate_layers(*, past_key_values, **_):
        seen["source_seq_len"] = past_key_values[0][0].shape[2]
        return past_key_values

    monkeypatch.setattr(lsc_eval, "translate_layers", fake_translate_layers)

    artifacts = lsc_eval._build_logit_edge_artifacts(
        ctx=ctx,
        edge=edge,
        source_context_token_ids=source_tokens,
        target_context_token_ids=receiver_tokens,
        past_by_node_id={"A": mismatched_source_past, "B": native_target_past},
        translator_pool=object(),
    )

    assert mismatched_source_past[0][0].shape[2] == 5
    assert native_target_past[0][0].shape[2] == 3
    assert seen["source_seq_len"] == 3
    assert artifacts.translated_past_key_values[0][0].shape == native_target_past[0][0].shape
    assert torch.isfinite(torch.tensor(artifacts.cosine_value))


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


class _FloatEmbedding(torch.nn.Module):
    def forward(self, token_ids):
        return token_ids.to(torch.float32).unsqueeze(-1)


def test_mot_retain_rotary_inputs_keep_absolute_positions_and_prefix_mask() -> None:
    decoder = SimpleNamespace(embed_tokens=_FloatEmbedding())
    token_ids = TokenIDs(torch.tensor([[11, 12]], dtype=torch.long), model_id="tiny")
    hidden, position_ids, attention_mask, position_embeddings = _build_rotary_input_hidden_states(
        decoder,
        token_ids,
        retain_prefix_length=3,
    )

    assert position_embeddings is None
    assert position_ids.tolist() == [[3, 4]]
    assert tuple(attention_mask.shape) == (1, 1, 2, 5)
    # Query at absolute position 3 can see prefix 0..2 and itself, but not pos 4.
    assert torch.all(attention_mask[0, 0, 0, :4] == 0)
    assert attention_mask[0, 0, 0, 4] < 0
    # Query at absolute position 4 can see all existing keys.
    assert torch.all(attention_mask[0, 0, 1] == 0)


def test_mot_retain_sparse_topk_uses_full_source_but_replays_delta_queries(monkeypatch) -> None:
    import alg.mot.train as mot_train

    source_delta = TokenIDs(torch.tensor([[30, 31]], dtype=torch.long), model_id="same")
    target_delta = TokenIDs(torch.tensor([[30, 31]], dtype=torch.long), model_id="same")
    source_full = TokenIDs(torch.tensor([[10, 11, 12, 30, 31]], dtype=torch.long), model_id="same")
    prefix_past = tuple(
        (
            torch.ones(1, 1, 3, 2),
            torch.ones(1, 1, 3, 2),
        )
        for _ in range(2)
    )
    source_delta_past = tuple(
        (
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
        )
        for _ in range(2)
    )
    source_full_past = tuple(
        (
            torch.zeros(1, 1, 5, 2),
            torch.zeros(1, 1, 5, 2),
        )
        for _ in range(2)
    )

    source_model = SimpleNamespace(id="same")
    target_model = SimpleNamespace(id="same")
    fake_ctx = SimpleNamespace(
        nodes=[SimpleNamespace(id="A", model_id="same"), SimpleNamespace(id="B", model_id="same")],
        tp=SimpleNamespace(get_model_spec=lambda _node: ModelSpec(model_id="same", num_layers=2, hidden_size=2, num_heads=1, head_dim=2)),
        cm=SimpleNamespace(
            get_src_layer_indices=lambda _edge: [0],
            get_tgt_layer_indices=lambda _edge: [0],
        ),
        config=SimpleNamespace(topk_sparse_attn=2, num_bottom_full_attn=0),
    )

    seen = {}

    def fake_sparse(model, context_token_ids, **kwargs):
        del model, kwargs
        seen["sparse_context"] = context_token_ids.as_tensor().clone()
        # Full-context query rows 0..4. Keep absolute key indices deliberately.
        indices = torch.tensor([[[[0, 0], [0, 1], [1, 2], [0, 3], [1, 4]]]], dtype=torch.long)
        return [indices.clone(), indices.clone()]

    def fake_translate(**kwargs):
        past = kwargs["past_key_values"]
        seq_len = past[0][0].shape[2]
        seen["translated_source_tokens"] = seq_len
        # [batch, seq, translated_layers, hidden]
        return torch.full((1, seq_len, 1, 2), 7.0), torch.full((1, seq_len, 1, 2), 8.0)

    def fake_replay(**kwargs):
        seen["replay_sparse"] = kwargs["sparse_attention_indices"]
        seen["replay_prefix"] = kwargs["retain_target_prefix_past_key_values"]
        seen["replay_injected_prefix"] = kwargs["retain_target_prefix_injected_window_past_key_values"]
        return source_delta_past

    monkeypatch.setattr(mot_train, "build_extrapolated_sparse_attention_indices", fake_sparse)
    monkeypatch.setattr(mot_train, "translate_layer_window", fake_translate)
    monkeypatch.setattr(mot_train, "replay_target_prefill_with_injected_window", fake_replay)

    replayed, _ = build_replayed_target_past(
        fake_ctx,
        source_past_key_values=source_delta_past,
        source_context_token_ids=source_delta,
        target_context_token_ids=target_delta,
        source_model=source_model,
        target_model=target_model,
        src_node_id="A",
        tgt_node_id="B",
        tgt_spec=ModelSpec(model_id="same", num_layers=2, hidden_size=2, num_heads=1, head_dim=2),
        retain_source_full_context_token_ids=source_full,
        retain_source_full_past_key_values=source_full_past,
        retain_target_prefix_past_key_values=prefix_past,
    )

    assert torch.equal(seen["sparse_context"], source_full.as_tensor())
    assert seen["translated_source_tokens"] == 5
    # Only delta query rows (absolute positions 3,4) remain; key indices stay absolute.
    assert seen["replay_sparse"][0].tolist() == [[[[0, 3], [1, 4]]]]
    assert seen["replay_prefix"] is prefix_past
    assert seen["replay_injected_prefix"][0][0].shape[2] == 3
    assert torch.all(seen["replay_prefix"][0][0] == 1.0)
    assert torch.all(seen["replay_injected_prefix"][0][0] == 7.0)
    assert torch.all(seen["replay_injected_prefix"][0][1] == 8.0)
    assert replayed is source_delta_past


def test_mot_retain_replay_uses_translated_prefix_only_at_injected_layers(monkeypatch) -> None:
    import alg.mot.train as mot_train

    target_model = SimpleNamespace(id="same")
    context_ids = TokenIDs(torch.tensor([[30, 31]], dtype=torch.long), model_id="same")
    tgt_spec = ModelSpec(model_id="same", num_layers=2, hidden_size=2, num_heads=1, head_dim=2)
    native_prefix = (
        (torch.full((1, 1, 3, 2), 10.0), torch.full((1, 1, 3, 2), 10.5)),
        (torch.full((1, 1, 3, 2), 11.0), torch.full((1, 1, 3, 2), 11.5)),
    )
    translated_prefix = (
        (torch.full((1, 1, 3, 2), 99.0), torch.full((1, 1, 3, 2), 98.0)),
    )
    injected_key = torch.full((1, 2, 1, 2), 7.0)
    injected_value = torch.full((1, 2, 1, 2), 8.0)
    layer0, layer1 = object(), object()
    decoder = SimpleNamespace(layers=[layer0, layer1])
    seen = []

    monkeypatch.setattr(mot_train, "resolve_target_model_family", lambda *args, **kwargs: "qwen2")
    monkeypatch.setattr(mot_train, "require_qwen2_model", lambda _model: decoder)

    def fake_inputs(_model, _tokens, retain_prefix_length=None):
        assert retain_prefix_length == 3
        hidden = torch.zeros(1, 2, 2)
        pos = torch.tensor([[3, 4]], dtype=torch.long)
        mask = torch.zeros(1, 1, 2, 5)
        return hidden, pos, mask, None

    def fake_run(block, hidden_states, **kwargs):
        prefix = kwargs["retain_prefix_present"]
        seen.append((block, float(prefix[0].mean().item()), float(prefix[1].mean().item())))
        present = (torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))
        return hidden_states, present

    monkeypatch.setattr(mot_train, "build_qwen2_input_hidden_states", fake_inputs)
    monkeypatch.setattr(mot_train, "run_qwen2_block", fake_run)

    replayed = mot_train.replay_target_prefill_with_injected_window(
        target_model=target_model,
        context_token_ids=context_ids,
        target_layer_indices=[1],
        injected_key_block=injected_key,
        injected_value_block=injected_value,
        tgt_spec=tgt_spec,
        sparse_attention_indices=None,
        num_bottom_full_attn=0,
        retain_target_prefix_past_key_values=native_prefix,
        retain_target_prefix_injected_window_past_key_values=translated_prefix,
    )

    assert len(replayed) == 2
    assert seen[0] == (layer0, 10.0, 10.5)
    assert seen[1] == (layer1, 99.0, 98.0)


def test_mot_retain_cross_append_aligns_full_history_but_replays_only_target_delta(monkeypatch) -> None:
    import alg.mot.train as mot_train

    source_model = SimpleNamespace(id="src")
    target_model = SimpleNamespace(id="tgt")
    src_spec = SimpleNamespace(num_layers=1)
    tgt_spec = SimpleNamespace(num_layers=1, num_key_value_heads=1, head_dim=1)
    ctx = SimpleNamespace(
        nodes=[SimpleNamespace(id="S", model_id="src"), SimpleNamespace(id="T", model_id="tgt")],
        tp=SimpleNamespace(get_model_spec=lambda node_id: src_spec if node_id == "S" else tgt_spec),
        cm=SimpleNamespace(
            get_src_layer_indices=lambda edge_id: [0],
            get_tgt_layer_indices=lambda edge_id: [0],
        ),
        config=SimpleNamespace(topk_sparse_attn=1, num_bottom_full_attn=0),
    )

    source_full_ids = TokenIDs(torch.tensor([[10, 11, 12, 13, 14]]), model_id="src")
    source_delta_ids = TokenIDs(torch.tensor([[13, 14]]), model_id="src")
    target_prefix_ids = [20, 21]
    target_delta_ids = TokenIDs(torch.tensor([[22, 23, 24]]), model_id="tgt")
    target_full_ids = TokenIDs(torch.tensor([[20, 21, 22, 23, 24]]), model_id="tgt")

    def fake_past(tokens):
        k = torch.zeros(1, 1, tokens, 1)
        v = torch.zeros(1, 1, tokens, 1)
        return ((k, v),)

    source_full_past = fake_past(5)
    source_delta_past = fake_past(2)
    target_prefix_past = fake_past(2)
    seen = {}

    monkeypatch.setattr(
        mot_train,
        "build_extrapolated_sparse_attention_indices",
        lambda *args, **kwargs: [torch.arange(5).view(1, 1, 5, 1)],
    )

    def fake_alignment(*, source_context_token_ids, target_context_token_ids, **kwargs):
        seen["alignment_source_len"] = int(source_context_token_ids.shape[1])
        seen["alignment_target_len"] = int(target_context_token_ids.shape[1])
        weights = torch.eye(5).unsqueeze(0)
        mapping = torch.arange(5).unsqueeze(0)
        return weights, mapping, mapping

    monkeypatch.setattr(mot_train, "build_cross_token_alignment", fake_alignment)

    def fake_align_sparse(indices, *, target_to_source, source_to_target):
        del target_to_source, source_to_target
        # Full target-grid sparse rows; production code must slice the two-token
        # resident target prefix and replay only rows for [22,23,24].
        return indices.clone()

    monkeypatch.setattr(mot_train, "align_sparse_attention_indices_to_target_tokens", fake_align_sparse)

    def fake_translate(*, past_key_values, token_alignment_weights=None, **kwargs):
        seen["translated_source_tokens"] = past_key_values[0][0].shape[2]
        seen["alignment_shape"] = tuple(token_alignment_weights.shape)
        key = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1)
        return key, key.clone()

    monkeypatch.setattr(mot_train, "translate_layer_window", fake_translate)

    def fake_blocks(*, key_block, value_block, **kwargs):
        del kwargs
        assert key_block.shape == value_block.shape
        k = key_block.permute(0, 2, 1, 3).contiguous()
        v = value_block.permute(0, 2, 1, 3).contiguous()
        return ((k, v),)

    monkeypatch.setattr(mot_train, "blocks_to_partial_past_key_values", fake_blocks)

    def fake_replay(*, context_token_ids, injected_key_block, sparse_attention_indices,
                    retain_target_prefix_past_key_values,
                    retain_target_prefix_injected_window_past_key_values, **kwargs):
        seen["replay_delta_tokens"] = int(context_token_ids.shape[1])
        seen["injected_delta_tokens"] = int(injected_key_block.shape[1])
        seen["sparse_delta_rows"] = int(sparse_attention_indices[0].shape[2])
        seen["native_prefix_tokens"] = retain_target_prefix_past_key_values[0][0].shape[2]
        seen["translated_prefix_tokens"] = retain_target_prefix_injected_window_past_key_values[0][0].shape[2]
        return fake_past(int(context_token_ids.shape[1]))

    monkeypatch.setattr(mot_train, "replay_target_prefill_with_injected_window", fake_replay)

    mixed, translated = build_replayed_target_past(
        ctx,
        source_past_key_values=source_delta_past,
        source_context_token_ids=source_delta_ids,
        target_context_token_ids=target_delta_ids,
        source_model=source_model,
        target_model=target_model,
        src_node_id="S",
        tgt_node_id="T",
        tgt_spec=tgt_spec,
        retain_source_full_context_token_ids=source_full_ids,
        retain_source_full_past_key_values=source_full_past,
        retain_target_prefix_past_key_values=target_prefix_past,
        retain_target_full_context_token_ids=target_full_ids,
        retain_source_prefix_tokens=3,
    )

    assert mixed[0][0].shape[2] == 3
    assert translated[0][0].shape[2] == 3
    assert seen == {
        "alignment_source_len": 5,
        "alignment_target_len": 5,
        "translated_source_tokens": 5,
        "alignment_shape": (1, 5, 5),
        "replay_delta_tokens": 3,
        "injected_delta_tokens": 3,
        "sparse_delta_rows": 3,
        "native_prefix_tokens": 2,
        "translated_prefix_tokens": 2,
    }
