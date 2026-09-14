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


def test_agent_runner_expert_persona_prompt_is_generated_per_example() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=4,
        log_turns=False,
    )

    class CapturingTokenizer:
        chat_template = "capture-template"

        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "rendered-expert-persona"

    tokenizer = CapturingTokenizer()
    original_tokenizer = runner.hub_agent.model.tokenizer
    runner.hub_agent.model.tokenizer = tokenizer
    try:
        assert runner.agent_personas == {}
        prompt = runner._build_expert_persona_prompt(
            context="premise",
            question="hypothesis",
            existing_personas=(("Semantic Expert", "Checks meaning."),),
        )
    finally:
        runner.hub_agent.model.tokenizer = original_tokenizer

    assert prompt == "rendered-expert-persona"
    assert [message["role"] for message in tokenizer.messages] == ["system", "user", "system", "user"]
    assert "one participant at a time" in tokenizer.messages[0]["content"]
    assert "complementing the existing participants" in tokenizer.messages[0]["content"]
    assert "Example 1:" in tokenizer.messages[0]["content"]
    assert "Example 2:" in tokenizer.messages[0]["content"]
    assert "Example 3:" in tokenizer.messages[0]["content"]
    assert "Now generate a participant" in tokenizer.messages[1]["content"]
    assert "Context:\npremise hypothesis" in tokenizer.messages[1]["content"]
    assert "Already Generated Participants" in tokenizer.messages[2]["content"]
    assert "Semantic Expert" in tokenizer.messages[2]["content"]
    assert "Only answer with the JSON for the next persona" in tokenizer.messages[3]["content"]
    assert AgentRunner._parse_persona('{"role":"Reviewer","description":"Checks errors."}', 1) == (
        "Reviewer",
        "Checks errors.",
    )
    assert tokenizer.messages[0]["content"].startswith("\nWhen faced with a task")
    assert AgentRunner._try_parse_persona('{"name":"Reviewer","description":"Checks errors."}') is None

def test_agent_runner_defaults_to_mallm_sampling_temperature() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        log_turns=False,
    )

    assert runner.generation_temperature == 1.0
    assert [agent.temperature for agent in runner.agent_sequence] == [1.0, 1.0]


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


def test_mallm_turn_renderer_preserves_system_and_user_roles_for_memory_continuation() -> None:
    class LlamaLikeTokenizer:
        chat_template = "llama-template"
        bos_token = "<|begin_of_text|>"

        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "<|begin_of_text|><system>general debate</system><user>simple</user><assistant>"

    tokenizer = LlamaLikeTokenizer()
    agent = SimpleNamespace(model=SimpleNamespace(id="meta-llama/Llama-3.2-1B-Instruct", tokenizer=tokenizer))

    rendered = AgentRunner._render_mallm_turn_prompt(
        agent,
        "general debate",
        ["simple", "Let's think step by step."],
        continuation=True,
    )

    assert tokenizer.messages == [
        {"role": "system", "content": "general debate"},
        {"role": "user", "content": "simple"},
        {"role": "user", "content": "Let's think step by step."},
    ]
    assert not rendered.startswith(tokenizer.bos_token)
    assert rendered.startswith("<system>general debate")


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


def test_agent_runner_final_answer_parser_extracts_anli_label() -> None:
    transcript = "Earlier text says FINAL: neutral, but only the final response should be parsed.\n"

    assert AgentRunner.extract_final_answer(transcript, "FINAL: ENTAILMENT") == "entailment"
    assert AgentRunner.extract_final_answer(transcript, "Reasoning... Label: contradiction") == "contradiction"


def test_agent_runner_persists_mallm_discussion_field_in_shared_memory() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        log_turns=False,
    )

    first_memory = runner._render_memory_entry(
        agent=runner.agents["A"],
        persona=("Semantic Expert", "Checks meaning."),
        response="Label: neutral. Reasoning.",
        context="premise",
        question="hypothesis",
        include_base=True,
    )
    later_memory = runner._render_memory_entry(
        agent=runner.agents["B"],
        persona=("Logic Reviewer", "Checks inference."),
        response="[AGREE]",
        context="premise",
        question="hypothesis",
        include_base=False,
    )

    assert first_memory.startswith("This is the discussion to the current point:")
    assert "Semantic Expert: Label: neutral. Reasoning." in first_memory
    assert "This is the discussion to the current point:" not in later_memory
    assert "Logic Reviewer: [AGREE]" in later_memory
    assert "Current Solution:" not in later_memory
    assert "Improve the current solution." not in later_memory


def test_agent_runner_memory_uses_official_header_and_nonself_user_role() -> None:
    class CapturingTokenizer:
        chat_template = "capture-template"
        bos_token = "<bos>"

        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "<bos><system>This is the discussion to the current point:</system>"

    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        log_turns=False,
    )
    tokenizer = CapturingTokenizer()
    fake_agent = SimpleNamespace(model=SimpleNamespace(tokenizer=tokenizer))
    rendered = runner._render_memory_entry(
        agent=fake_agent,
        persona=("Semantic Expert", "Checks meaning."),
        response="Label: neutral",
        context="premise",
        question="hypothesis",
        include_base=True,
    )

    assert [message["role"] for message in tokenizer.messages] == ["system", "user"]
    assert tokenizer.messages[0]["content"] == "This is the discussion to the current point: "
    assert tokenizer.messages[1]["content"] == "Semantic Expert: Label: neutral"
    assert rendered.startswith("<bos><system>")

def test_agent_runner_prompts_match_kv_native_mallm_memory_simple_structure() -> None:
    initial = AgentRunner._build_initial_user_content(
        "premise",
        "hypothesis",
        persona=("Semantic Expert", "Checks meaning."),
    )
    ordinary = AgentRunner._build_followup_user_content(
        "hypothesis",
        context="premise",
        persona=("Adversarial Reviewer", "Find missed evidence."),
        current_solution="neutral",
    )
    plain_initial = AgentRunner._build_plain_initial_prompt(
        "premise",
        "hypothesis",
        persona=("Semantic Expert", "Checks meaning."),
    )

    assert initial.startswith("You take part in a discussion to solve a task.")
    assert "Task:" in initial
    assert "Context:\npremise" in initial
    assert "Input: hypothesis" in initial
    assert "Your role: Semantic Expert (Checks meaning.)" in initial
    assert "Nobody proposed a solution yet." not in initial
    assert "Current Solution:" not in initial
    assert "This is the discussion to the current point:" not in initial

    assert ordinary.startswith("You take part in a discussion to solve a task.")
    assert "Context:\npremise" in ordinary
    assert "Input: hypothesis" in ordinary
    assert "Your role: Adversarial Reviewer (Find missed evidence.)" in ordinary
    assert "Current Solution: neutral" in ordinary
    # The official Memory header lives in the reusable KV prefix in AgentRunner
    # because causal KV reuse requires the history to precede dynamic control.
    assert "This is the discussion to the current point:" not in ordinary
    assert (
        "Improve the current solution. If you agree with the current solution, answer with [AGREE], "
        "else answer with [DISAGREE] and explain why and provide an improved solution."
    ) in ordinary
    assert "Let's think step by step." in ordinary

    assert "Propose a solution." in plain_initial
    assert "Improve the current solution." not in plain_initial
    assert "Let's think step by step." in plain_initial

def test_agent_runner_solution_extraction_matches_mallm_system_user_roles() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        log_turns=False,
    )
    runner.agent_personas["A"] = ("Semantic Expert", "Checks meaning.")

    class CapturingTokenizer:
        chat_template = "capture-template"

        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "rendered-extraction"

    tokenizer = CapturingTokenizer()
    fake_agent = SimpleNamespace(
        node_id="A",
        model=SimpleNamespace(tokenizer=tokenizer),
    )
    prompt = runner._build_solution_extraction_prompt(
        agent=fake_agent,
        context="premise",
        question="hypothesis",
        response="[DISAGREE] entailment because ...",
    )

    assert prompt == "rendered-extraction"
    assert [message["role"] for message in tokenizer.messages] == ["system", "user", "user"]
    assert tokenizer.messages[0]["content"] == (
        "You are tasked with creating a final solution based on the given input and your previous response."
    )
    user_content = tokenizer.messages[1]["content"]
    assert "Task:" in user_content
    assert "Context:\npremise" in user_content
    assert "Input: hypothesis" in user_content
    assert "Your previous response: [DISAGREE] entailment because ..." in user_content
    assert "Semantic Expert" not in "\n".join(message["content"] for message in tokenizer.messages)
    assert "Extract the final solution to the task from the provided text." in tokenizer.messages[2]["content"]


def test_agent_runner_anli_task_instruction_does_not_force_custom_output_format() -> None:
    instruction = AgentRunner._task_instruction()

    assert "entailment" in instruction
    assert "neutral" in instruction
    assert "contradiction" in instruction
    assert "Label:" not in instruction
    assert "before the explanation" not in instruction

def test_agent_runner_preserves_mallm_extracted_solution_as_current_draft() -> None:
    response = (
        "[DISAGREE] The old Label: neutral misses a geographic implication. "
        "The relationship should be classified as entailment because Brazil is in South America."
    )
    extracted_solution = "The relationship is entailment"

    solution, label, state = AgentRunner._response_updates_solution(
        response,
        current_solution="neutral",
        current_label="neutral",
        extracted_solution=extracted_solution,
    )

    # FreeTextResponseGenerator keeps extract_result() verbatim in Response.solution.
    assert solution == extracted_solution
    assert label == "entailment"
    assert state == "revise"
    assert response not in solution


def test_agent_runner_anli_label_parser_normalizes_common_variants() -> None:
    assert AgentRunner._extract_anli_label("Label: entailed\nReason") == "entailment"
    assert AgentRunner._extract_anli_label("Label: contradictory\nReason") == "contradiction"
    assert AgentRunner._extract_anli_label(
        "[DISAGREE] The old Label: neutral is wrong. The relationship is classified as entailment."
    ) == "entailment"


def test_agent_runner_majority_consensus_tracks_agreement_on_current_solution() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=4,
        log_turns=False,
    )

    decision, recent = runner._majority_consensus([(False, "entailment")])
    assert decision is None
    assert recent == [(False, "entailment")]

    # Match the official ThresholdConsensus implementation literally: agreeing
    # Panelists carry the previous truthy solution, so the reverse scan still
    # finds a solution at position 1 and does not accumulate a 2/4 majority.
    decision, _ = runner._majority_consensus(
        [(False, "entailment"), (True, "entailment")]
    )
    assert decision is None

    # If the newest agreement has no solution, the previous proposal is at
    # reverse position 2 and exactly 2/4 satisfies the official >= 0.5 check.
    decision, _ = runner._majority_consensus(
        [(False, "entailment"), (True, "")]
    )
    assert decision == "entailment"

    two_agent_runner = AgentRunner(
        ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=2, log_turns=False
    )
    decision, _ = two_agent_runner._majority_consensus([(False, "entailment")])
    assert decision == "entailment"

    solution, label, state = AgentRunner._response_updates_solution(
        "[AGREE]",
        current_solution="Label: neutral because ...",
        current_label="neutral",
    )
    assert solution == "Label: neutral because ..."
    assert label == "neutral"
    assert state == "agree"

    solution, label, state = AgentRunner._response_updates_solution(
        "I agree with the current solution.",
        current_solution="Label: neutral because ...",
        current_label="neutral",
    )
    assert solution == "Label: neutral because ..."
    assert label == "neutral"
    assert state == "agree"

    solution, label, state = AgentRunner._response_updates_solution(
        "[DISAGREE] Different reasoning, but still neutral. Label: neutral",
        current_solution="Label: neutral because ...",
        current_label="neutral",
        extracted_solution="neutral",
    )
    assert label == "neutral"
    assert solution == "neutral"
    assert state == "revise"

    solution, label, state = AgentRunner._response_updates_solution(
        "[DISAGREE] The premise rules it out. Label: contradiction",
        current_solution="Label: neutral because ...",
        current_label="neutral",
        extracted_solution="contradiction",
    )
    assert label == "contradiction"
    assert solution == "contradiction"
    assert state == "revise"

    solution, label, state = AgentRunner._response_updates_solution(
        "[DISAGREE] I cannot support the current draft.",
        current_solution="Label: neutral because ...",
        current_label="neutral",
        extracted_solution="",
    )
    assert solution == ""
    assert label is None
    assert state == "revise"

def test_multi_agents_qa_defaults_to_anli_r3_dev() -> None:
    from exp.multi_agents_qa import build_parser

    args = build_parser().parse_args(["mot", "--checkpoint-dir-path", "checkpoint"])
    assert args.split == "dev_r3"
    assert args.data_dir == "./anli"
    assert args.max_turns == 7
    assert args.generation_max_new_tokens == 1024
    assert args.generation_temperature == 1.0
    assert args.agent_count == 3



def test_agent_runner_stops_early_when_majority_consensus_is_reached() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        max_turns=4,
        log_turns=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._extract_solution_with_model = lambda *, agent, context, question, response: AgentRunner._solution_from_response(response)

    calls = []
    responses = {
        "A": "Label: neutral because ...",
        "B": "[AGREE]",
    }

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls.append(agent.node_id)
            text = responses[agent.node_id]
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
    result = runner.run(context="premise", question="hypothesis", gold_answers=["neutral"])

    assert calls == ["A"]
    assert len(result.turns) == 1
    assert result.prediction == "neutral"
    assert result.profile["consensus_reached"] is True
    assert result.profile["consensus_turn"] == 1
    assert result.profile["persona_generator"] == "expert"
    assert result.profile["response_generator"] == "simple"
    assert result.profile["discussion_paradigm"] == "memory"
    assert result.profile["decision_protocol"] == "majority_consensus"
    assert result.profile["use_chain_of_thought"] is True
    assert result.profile["requested_max_turns"] == 4
    assert result.profile["num_agent_turns"] == 1


def test_agent_runner_max_turns_counts_mallm_rounds() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=3,
        max_turns=2,
        log_turns=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._extract_solution_with_model = lambda *, agent, context, question, response: AgentRunner._solution_from_response(response)
    calls = []
    responses = {
        "A": "[DISAGREE] Label: entailment",
        "B": "[DISAGREE] Label: contradiction",
        "C": "[DISAGREE] Label: neutral",
    }

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls.append(agent.node_id)
            text = responses[agent.node_id]
            return AgentGeneration(
                agent_id=agent.node_id, prompt_text=prompt_text, text=text, raw_text=text,
                generated_token_ids=[1], tokens_before=0, tokens_after=1,
                tokens_prompt=1, tokens_completion=1,
            )
        return generate

    for agent in runner.agent_sequence:
        agent.generate_response = fake_generate(agent)
    runner._prepare_outgoing_route_translation = lambda **kwargs: None
    runner._update_peak_memory_breakdown = lambda: None

    def fake_star_offload(*, source_agent, target_agent):
        meta = {"edge_id": "edge", "offload_kind": "delta", "tokens_sent": 1,
                "tokens_received": 1, "expected_delta_tokens": 1}
        return target_agent, meta, False, source_agent, meta

    runner._star_offload_to_turn_target = fake_star_offload
    result = runner.run(context="premise", question="hypothesis", gold_answers=["contradiction"])

    assert calls == ["A", "B", "C", "A", "B", "C"]
    assert len(result.turns) == 6
    assert result.profile["consensus_reached"] is False
    assert result.profile["num_agent_turns"] == 6

def test_agent_runner_disagreement_resets_consensus_even_when_label_is_unchanged() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=5,
        max_turns=4,
        log_turns=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._extract_solution_with_model = lambda *, agent, context, question, response: AgentRunner._solution_from_response(response)

    responses = {
        "A": "Label: neutral. First draft.",
        "B": "[AGREE]",
        "C": "[DISAGREE] Label: neutral. Revised reasoning.",
        "D": "[AGREE]",
        "E": "[DISAGREE] Label: neutral. Another revision.",
    }

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            text = responses[agent.node_id]
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
    result = runner.run(context="premise", question="hypothesis", gold_answers=["neutral"])

    assert len(result.turns) == 20
    assert result.prediction == "neutral"
    assert result.profile["consensus_reached"] is False


def test_agent_runner_returns_current_solution_when_max_turns_end_without_consensus() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=4,
        max_turns=4,
        log_turns=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._extract_solution_with_model = lambda *, agent, context, question, response: AgentRunner._solution_from_response(response)

    responses = {
        "A": "Label: entailment",
        "B": "[DISAGREE] Label: contradiction",
        "C": "[DISAGREE] Label: neutral",
        "D": "[DISAGREE] Label: contradiction",
    }
    calls = []

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls.append(agent.node_id)
            text = responses[agent.node_id]
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
    result = runner.run(context="premise", question="hypothesis", gold_answers=["neutral"])

    assert calls == ["A", "B", "C", "D"] * 4
    assert len(result.turns) == 16
    assert result.prediction == "contradiction"
    assert result.profile["consensus_reached"] is False
    assert result.profile["consensus_turn"] is None



def test_agent_runner_strips_llama_date_metadata_from_rendered_prompt() -> None:
    rendered = (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\n"
        "Today Date: 26 Jul 2024\n\n"
        "You take part in a discussion to solve a task."
    )

    cleaned = AgentRunner._strip_irrelevant_chat_template_metadata(rendered)

    assert "Cutting Knowledge Date" not in cleaned
    assert "Today Date" not in cleaned
    assert "<|start_header_id|>system<|end_header_id|>" in cleaned
    assert "You take part in a discussion to solve a task." in cleaned
