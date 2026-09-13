from types import SimpleNamespace

import pytest

from core.agent import Agent, AgentGeneration
from core.agent_runner import AgentRunner, AgentTurnRecord
from core.context import Context


def _render_qwen_chat_prompt(agent, user_content: str):
    """Test-only alias used to exercise the generic chat renderer with Qwen fixtures."""
    return AgentRunner._render_chat_prompt(agent, user_content, continuation=False)


def _ctx(model_ids: str) -> Context:
    return Context(
        SimpleNamespace(
            model_ids=model_ids,
            model_directions="A_to_B",
            device="cpu",
            dtype="float32",
        )
    )


def test_agent_encode_text_does_not_auto_insert_special_tokens() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    model = ctx.tp.get_model("A")
    agent = Agent(node_id="A", model=model, device="cpu")

    encoded = agent.encode_text("abcd")
    expected = model.tokenizer("abcd", add_special_tokens=False).input_ids

    assert encoded.squeeze(0).tolist() == expected
    assert model.tokenizer.eos_token_id not in encoded.squeeze(0).tolist()


def test_agent_collects_all_configured_eos_token_ids() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    model = ctx.tp.get_model("A")
    model.model.generation_config = SimpleNamespace(eos_token_id=[7, 9])
    agent = Agent(node_id="A", model=model, device="cpu")

    assert agent._eos_token_ids() == {model.tokenizer.eos_token_id, 7, 9}


def test_agent_runner_virtual_agents_share_one_physical_model() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=6,
        log_turns=False,
    )

    assert len(runner.agent_sequence) == 6
    assert len({id(agent.model) for agent in runner.agent_sequence}) == 1
    assert runner.agent_sequence[0].model is ctx.tp.get_model("A")
    assert ctx.tp.get_model("A") is ctx.tp.get_model("B")


def test_agent_runner_rejects_heterogeneous_model_pool() -> None:
    ctx = _ctx("tiny-a,tiny-b")

    with pytest.raises(ValueError, match="homogeneous-model control experiment"):
        AgentRunner(
            ctx=ctx,
            translator_pool=ctx.tp,
            alg="mot",
            agent_count=2,
            log_turns=False,
        )


def test_agent_runner_rejects_heterogeneous_checkpoint_before_loading_models(tmp_path, monkeypatch) -> None:
    import json
    from core.agent_runner import AgentRunnerConfig

    (tmp_path / "train_config.json").write_text(
        json.dumps(
            {
                "model_ids": "tiny-a,tiny-b",
                "model_directions": "A_to_B",
            }
        )
    )

    def fail_if_loader_is_reached(*args, **kwargs):
        raise AssertionError("checkpoint loader should not be reached for heterogeneous AgentRunner control")

    monkeypatch.setattr("core.agent_runner.importlib.import_module", fail_if_loader_is_reached)

    with pytest.raises(ValueError, match="heterogeneous checkpoints"):
        AgentRunner.from_checkpoint(
            AgentRunnerConfig(
                alg="mot",
                checkpoint_dir_path=str(tmp_path),
                device="cpu",
            )
        )


def test_qwen_prompt_uses_checkpoint_template_without_family_specific_overrides() -> None:
    class ChatTokenizer:
        chat_template = "checkpoint-template"

        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            return "rendered-chat"

    tokenizer = ChatTokenizer()
    fake_agent = SimpleNamespace(
        model=SimpleNamespace(
            id="Qwen/Qwen3-4B-Instruct-2507",
            config=SimpleNamespace(model_type="qwen3"),
            tokenizer=tokenizer,
        )
    )

    rendered = _render_qwen_chat_prompt(fake_agent, "question")

    assert rendered == "rendered-chat"
    assert tokenizer.kwargs == {
        "chat_template": "checkpoint-template",
        "tokenize": False,
        "add_generation_prompt": True,
    }


def test_chat_template_is_used_for_non_qwen_and_falls_back_when_system_role_is_unsupported() -> None:
    class GemmaLikeTokenizer:
        chat_template = "gemma-template"
        bos_token = "<bos>"

        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if messages[0]["role"] == "system":
                raise ValueError("system role unsupported")
            return "<bos><start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"

    tokenizer = GemmaLikeTokenizer()
    agent = SimpleNamespace(model=SimpleNamespace(id="google/gemma-3-1b-it", tokenizer=tokenizer))

    rendered = AgentRunner._render_chat_prompt(agent, "hello", continuation=False)

    assert rendered.startswith("<bos>")
    assert len(tokenizer.calls) == 2
    assert tokenizer.calls[0][0][0]["role"] == "system"
    assert tokenizer.calls[1][0][0]["role"] == "user"


def test_chat_continuation_uses_user_only_fragment_and_strips_leading_bos() -> None:
    class LlamaLikeTokenizer:
        chat_template = "llama-template"
        bos_token = "<|begin_of_text|>"

        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nquestion"

    tokenizer = LlamaLikeTokenizer()
    agent = SimpleNamespace(model=SimpleNamespace(id="meta-llama/Llama-3.2-1B-Instruct", tokenizer=tokenizer))

    rendered = AgentRunner._render_chat_prompt(agent, "question", continuation=True)

    assert tokenizer.messages == [{"role": "user", "content": "question"}]
    assert not rendered.startswith(tokenizer.bos_token)
    assert rendered.startswith("<|start_header_id|>user")


def test_agent_caches_generated_eos_as_turn_terminator() -> None:
    import torch

    class OneTokenTokenizer:
        eos_token_id = 9
        bos_token = None

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            assert add_special_tokens is False
            ids = torch.tensor([[5]], dtype=torch.long)
            return SimpleNamespace(input_ids=ids) if return_tensors == "pt" else SimpleNamespace(input_ids=[5])

        def decode(self, token_ids, skip_special_tokens=True):
            return ""

    class EosModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.id = "fake-chat"
            self.tokenizer = OneTokenTokenizer()
            self.config = SimpleNamespace(eos_token_id=9)
            self.generation_config = SimpleNamespace(eos_token_id=9)
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, input_ids, past_key_values=None, use_cache=True):
            del use_cache
            old_len = 0 if past_key_values is None else int(past_key_values[0][0].shape[2])
            new_len = old_len + int(input_ids.shape[1])
            key = torch.zeros(1, 1, new_len, 1)
            value = torch.zeros_like(key)
            logits = torch.zeros(1, input_ids.shape[1], 16)
            logits[..., 9] = 1.0
            return SimpleNamespace(past_key_values=((key, value),), logits=logits)

    agent = Agent(node_id="A", model=EosModel(), device="cpu", max_new_tokens=4)
    generation = agent.generate_response("x")

    assert generation.generated_token_ids == []
    assert agent.cache_token_ids == [5, 9]
    assert agent.cache_seq_len == 2


def test_agent_forces_chat_turn_terminator_when_max_new_tokens_is_reached() -> None:
    import torch

    class ChatTokenizer:
        chat_template = "chat-template"
        eos_token_id = 1
        bos_token = None

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            assert add_special_tokens is False
            ids = [9] if text == "<eot>" else [5]
            tensor = torch.tensor([ids], dtype=torch.long)
            return SimpleNamespace(input_ids=tensor) if return_tensors == "pt" else SimpleNamespace(input_ids=ids)

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"] is False
            return f"<user>probe</user><assistant>{messages[-1]['content']}<eot>"

        def decode(self, token_ids, skip_special_tokens=True):
            return "x" * len(token_ids)

    class NeverEosModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.id = "fake-chat"
            self.tokenizer = ChatTokenizer()
            self.config = SimpleNamespace(eos_token_id=[1, 9])
            self.generation_config = SimpleNamespace(eos_token_id=[1, 9])
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def forward(self, input_ids, past_key_values=None, use_cache=True):
            del use_cache
            old_len = 0 if past_key_values is None else int(past_key_values[0][0].shape[2])
            new_len = old_len + int(input_ids.shape[1])
            key = torch.zeros(1, 1, new_len, 1)
            value = torch.zeros_like(key)
            logits = torch.zeros(1, input_ids.shape[1], 16)
            logits[..., 7] = 1.0
            return SimpleNamespace(past_key_values=((key, value),), logits=logits)

    agent = Agent(node_id="A", model=NeverEosModel(), device="cpu", max_new_tokens=2)
    generation = agent.generate_response("prompt")

    assert generation.generated_token_ids == [7, 7]
    assert generation.tokens_completion == 2
    assert agent._turn_terminator_token_id() == 9
    assert agent.cache_token_ids == [5, 7, 7, 9]
    assert agent.cache_seq_len == 4


def test_agent_runner_final_answer_parser_extracts_strategyqa_binary_answer() -> None:
    transcript = "Earlier text says FINAL: no, but only the final response should be parsed.\n"

    assert AgentRunner.extract_final_answer(transcript, "FINAL: YES") == "yes"
    assert AgentRunner.extract_final_answer(transcript, "Reasoning... answer: no") == "no"


def test_agent_runner_prompts_use_mallm_memory_simple_and_judge() -> None:
    ordinary = AgentRunner._build_followup_user_content(
        "question", agent_id="B", previous_agent_id="A", is_final_turn=False
    )
    final = AgentRunner._build_followup_user_content(
        "question",
        agent_id="A",
        previous_agent_id="D",
        is_final_turn=True,
        judge_solutions=(("A", "Answer: yes"), ("B", "Answer: no")),
    )

    assert "FINAL:" not in ordinary
    assert "shared memory" in ordinary
    assert "full discussion history from all previous Agents" in ordinary
    assert "Improve the current solution" in ordinary
    assert "If you agree with the current solution" in ordinary
    assert "[AGREE] Answer: yes" in ordinary
    assert "[DISAGREE] Answer: no" in ordinary
    assert "improved solution" in ordinary
    assert "generation limit" in ordinary
    assert "Agent A" in ordinary
    assert "I agree with Agent [agent id]" not in ordinary

    assert "You are the Judge" in final
    assert "decision on the listed solutions" in final
    assert "### Solutions:" in final
    assert "Solution 1 (Agent A): Answer: yes" in final
    assert "Solution 2 (Agent B): Answer: no" in final
    assert "FINAL: yes" in final
    assert "FINAL: no" in final
    assert "Hub" not in final


def test_agent_runner_judge_chat_prompt_is_standalone_user_turn() -> None:
    class ChatTokenizer:
        chat_template = "checkpoint-template"

        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            self.messages = messages
            return "rendered-judge"

    tokenizer = ChatTokenizer()
    fake_agent = SimpleNamespace(model=SimpleNamespace(tokenizer=tokenizer))

    rendered = AgentRunner._render_chat_prompt(
        fake_agent,
        "judge input",
        continuation=False,
        include_system=False,
    )

    assert rendered == "rendered-judge"
    assert tokenizer.messages == [{"role": "user", "content": "judge input"}]


def test_agent_runner_judge_uses_latest_solution_per_agent() -> None:
    turns = [
        AgentTurnRecord("A", "", "A old", 0, 0, 0, 0),
        AgentTurnRecord("B", "", "B only", 0, 0, 0, 0),
        AgentTurnRecord("A", "", "A latest", 0, 0, 0, 0),
    ]

    assert AgentRunner._latest_agent_solutions(turns) == [("A", "A latest"), ("B", "B only")]


def test_multi_agents_qa_defaults_to_strategyqa_dev() -> None:
    from exp.multi_agents_qa import build_parser

    args = build_parser().parse_args(["mot", "--checkpoint-dir-path", "checkpoint"])
    assert args.split == "dev"
    assert args.data_dir == "./strategyqa"



def test_agent_runner_runs_exact_normal_turn_count_then_separate_final_hub_turn() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=4,
        max_turns=5,
        log_turns=False,
    )

    calls = []

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            is_final = "You are the Judge" in prompt_text
            calls.append((agent.node_id, is_final))
            text = "FINAL: yes" if is_final else f"ordinary-{agent.node_id}"
            return AgentGeneration(
                agent_id=agent.node_id,
                prompt_text=prompt_text,
                text=text,
                raw_text=text,
                generated_token_ids=[1],
                tokens_before=0,
                tokens_after=1,
                tokens_prompt=1,
                tokens_completion=1,
            )

        return generate

    for agent in runner.agent_sequence:
        agent.generate_response = fake_generate(agent)

    runner._prepare_outgoing_route_translation = lambda **kwargs: None
    runner._update_peak_memory_breakdown = lambda: None

    def fake_star_offload(*, source_agent, target_agent):
        meta = {
            "edge_id": "A_to_B",
            "offload_kind": "delta",
            "tokens_sent": 1,
            "tokens_received": 1,
            "expected_delta_tokens": 1,
        }
        return target_agent, meta, False, source_agent, meta

    runner._star_offload_to_turn_target = fake_star_offload

    result = runner.run(
        context="context",
        question="question",
        gold_answers=["yes"],
    )

    assert calls == [
        ("A", False),
        ("B", False),
        ("C", False),
        ("D", False),
        ("A", False),
        ("A", True),
    ]
    assert len(result.turns) == 6
    assert result.profile["requested_max_turns"] == 5
    assert result.profile["num_agent_turns"] == 6
    assert result.prediction == "yes"
    assert result.accuracy == 1.0


def test_agent_runner_judge_does_not_receive_discussion_kv_handoff() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=4,
        max_turns=4,
        log_turns=False,
    )

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            is_final = "You are the Judge" in prompt_text
            text = "FINAL: yes" if is_final else f"ordinary-{agent.node_id}"
            return AgentGeneration(
                agent_id=agent.node_id,
                prompt_text=prompt_text,
                text=text,
                raw_text=text,
                generated_token_ids=[1],
                tokens_before=0,
                tokens_after=1,
                tokens_prompt=1,
                tokens_completion=1,
            )

        return generate

    for agent in runner.agent_sequence:
        agent.generate_response = fake_generate(agent)

    runner._prepare_outgoing_route_translation = lambda **kwargs: None
    runner._update_peak_memory_breakdown = lambda: None
    offload_pairs = []

    def fake_star_offload(*, source_agent, target_agent):
        offload_pairs.append((source_agent.node_id, target_agent.node_id))
        meta = {
            "edge_id": "A_to_B",
            "offload_kind": "delta",
            "tokens_sent": 1,
            "tokens_received": 1,
            "expected_delta_tokens": 1,
        }
        return target_agent, meta, False, source_agent, meta

    runner._star_offload_to_turn_target = fake_star_offload

    result = runner.run(context="", question="question", gold_answers=["yes"])

    assert offload_pairs == [("A", "B"), ("B", "C"), ("C", "D")]
    assert result.turns[-1].translated_edge_id is None
    assert result.turns[-1].tokens_received == 0
    assert result.prediction == "yes"
