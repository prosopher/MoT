from types import SimpleNamespace

import pytest

from core.agent import Agent
from core.agent_runner import AgentRunner
from core.context import Context


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

    rendered = AgentRunner._render_qwen_chat_prompt(fake_agent, "question")

    assert rendered == "rendered-chat"
    assert tokenizer.kwargs == {
        "chat_template": "checkpoint-template",
        "tokenize": False,
        "add_generation_prompt": True,
    }
