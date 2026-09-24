from types import SimpleNamespace
import json

import pytest
import torch

from core.agent import Agent, AgentGeneration, get_past_seq_len
from core.agent_runner import AgentRunner, AgentMessageRecord
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
        log_agents=False,
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
        log_agents=False,
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
            agent=runner.hub_agent,
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
        log_agents=False,
    )

    assert runner.generation_temperature == 1.0
    assert [agent.temperature for agent in runner.agent_sequence] == [1.0, 1.0]


def test_agent_runner_generation_seed_streams_are_agent_count_independent(monkeypatch) -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner2 = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        seed=42,
        log_agents=False,
    )
    runner6 = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=6,
        seed=42,
        log_agents=False,
    )
    runner2._current_example_seed = 47
    runner6._current_example_seed = 47

    seen = []
    monkeypatch.setattr("core.agent_runner.set_seed", lambda seed: seen.append(seed))

    seeds2 = [
        runner2._set_generation_seed("persona", agent_index=0),
        runner2._set_generation_seed("discussion", turn_index=0, agent_index=0),
        runner2._set_generation_seed("discussion", turn_index=0, agent_index=1),
        runner2._set_generation_seed("discussion", turn_index=0, agent_index=1, attempt=1),
        runner2._set_generation_seed("verification", turn_index=0, agent_index=0),
        runner2._set_generation_seed("verification", turn_index=0, agent_index=1, attempt=1),
    ]
    seeds6 = [
        runner6._set_generation_seed("persona", agent_index=0),
        runner6._set_generation_seed("discussion", turn_index=0, agent_index=0),
        runner6._set_generation_seed("discussion", turn_index=0, agent_index=1),
        runner6._set_generation_seed("discussion", turn_index=0, agent_index=1, attempt=1),
        runner6._set_generation_seed("verification", turn_index=0, agent_index=0),
        runner6._set_generation_seed("verification", turn_index=0, agent_index=1, attempt=1),
    ]

    assert seeds2 == seeds6
    assert len(set(seeds2)) == len(seeds2)


def test_agent_runner_common_prefix_personas_are_agent_count_invariant(monkeypatch) -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner2 = AgentRunner(
        ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=2, seed=42, log_agents=False
    )
    runner6 = AgentRunner(
        ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=6, seed=42, log_agents=False
    )
    runner2._current_example_seed = 47
    runner6._current_example_seed = 47

    prompts_by_runner = {id(runner2): [], id(runner6): []}
    active_runner_id = [id(runner2)]

    def fake_generate(self, prompt_text: str):
        # _set_generation_seed() runs immediately before this call. Encoding that
        # seed into the persona makes this an end-to-end test of Agent-offset RNG
        # invariance, including the prompt's dependency on earlier personas.
        seed = int(torch.initial_seed())
        prompts_by_runner[active_runner_id[0]].append(prompt_text)
        text = json.dumps(
            {
                "role": f"Expert-{seed}",
                "description": f"Deterministic persona generated from seed {seed}.",
            }
        )
        return AgentGeneration(
            agent_id=self.node_id,
            prompt_text=prompt_text,
            text=text,
            raw_text=text,
            generated_token_ids=[],
            tokens_before=0,
            tokens_after=0,
            tokens_prompt=0,
            tokens_completion=0,
        )

    monkeypatch.setattr(Agent, "generate_response", fake_generate)

    active_runner_id[0] = id(runner2)
    personas2 = runner2._generate_expert_personas("premise", "hypothesis")
    active_runner_id[0] = id(runner6)
    personas6 = runner6._generate_expert_personas("premise", "hypothesis")

    assert list(personas2.items()) == list(personas6.items())[:2]
    assert prompts_by_runner[id(runner2)] == prompts_by_runner[id(runner6)][:2]


def test_agent_runner_stop_sequences_are_agent_count_invariant() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner2 = AgentRunner(
        ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=2, log_agents=False
    )
    runner8 = AgentRunner(
        ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=8, log_agents=False
    )

    assert runner2.agent_sequence[0].stop_sequences == runner8.agent_sequence[0].stop_sequences
    assert "\nAgent " in runner2.agent_sequence[0].stop_sequences
    assert not any(stop.startswith("\nAgent A:") for stop in runner2.agent_sequence[0].stop_sequences)


def test_agent_runner_seed_coordinates_do_not_alias_at_large_offsets(monkeypatch) -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=2, seed=42, log_agents=False
    )
    runner._current_example_seed = 142
    monkeypatch.setattr("core.agent_runner.set_seed", lambda seed: None)

    coordinates = [
        ("persona", 0, 0, 0),
        ("persona", 0, 0, 1),
        ("persona", 0, 100, 0),
        ("discussion", 0, 0, 0),
        ("discussion", 1, 0, 0),
        ("discussion", 0, 100, 0),
        ("verification", 0, 0, 0),
        ("verification", 100, 0, 0),
    ]
    seeds = [
        runner._set_generation_seed(
            stream, turn_index=turn_index, agent_index=agent_index, attempt=attempt
        )
        for stream, turn_index, agent_index, attempt in coordinates
    ]
    assert len(seeds) == len(set(seeds))


def test_agent_runner_retries_duplicate_or_corrupted_expert_personas() -> None:
    assert AgentRunner._persona_is_usable(
        ("Semantic Expert", "Checks meaning."),
        [("Logic Reviewer", "Checks inference.")],
    )
    assert not AgentRunner._persona_is_usable(
        ("Semantic Expert", "A duplicate role."),
        [("semantic   expert", "Checks meaning.")],
    )
    assert not AgentRunner._persona_is_usable(
        ("Programmer", "garbled output ```json"),
        [],
    )
    assert not AgentRunner._persona_is_usable(
        ("Researcher", "Corrupted persona with stray 幻 token."),
        [],
    )


def test_expert_persona_prompt_uses_generating_agents_chat_template() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=2, log_agents=False)

    class TaggedTokenizer:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.chat_template = f"template-{tag}"
            self.calls = 0
        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            assert kwargs["chat_template"] == self.chat_template
            return f"rendered-by-{self.tag}"

    hub_tokenizer = TaggedTokenizer("hub")
    target_tokenizer = TaggedTokenizer("target")
    original = runner.hub_agent.model.tokenizer
    runner.hub_agent.model.tokenizer = hub_tokenizer
    target_agent = SimpleNamespace(model=SimpleNamespace(tokenizer=target_tokenizer))
    try:
        prompt = runner._build_expert_persona_prompt(
            agent=target_agent, context="", question="Question?", existing_personas=()
        )
    finally:
        runner.hub_agent.model.tokenizer = original
    assert prompt == "rendered-by-target"
    assert target_tokenizer.calls == 1
    assert hub_tokenizer.calls == 0


def test_agent_runner_heterogeneous_virtual_agents_cycle_checkpoint_models() -> None:
    ctx = Context(
        SimpleNamespace(
            model_ids="tiny-a,tiny-b",
            model_directions="all",
            device="cpu",
            dtype="float32",
        )
    )
    runner = AgentRunner(ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=6, log_agents=False)
    assert [agent.model.id for agent in runner.agent_sequence] == [
        "tiny-a", "tiny-b", "tiny-a", "tiny-b", "tiny-a", "tiny-b"
    ]
    assert runner.logical_to_physical_node_id == {
        "A":"A", "B":"B", "C":"A", "D":"B", "E":"A", "F":"B"
    }


def test_agent_runner_rejects_heterogeneous_pool_for_algorithm_without_cross_tokenization() -> None:
    ctx = _ctx("tiny-a,tiny-b")
    with pytest.raises(ValueError, match="Heterogeneous AgentRunner is not defined"):
        AgentRunner(ctx=ctx, translator_pool=ctx.tp, alg="interlat", agent_count=2, log_agents=False)


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


def test_mallm_agent_prompt_renderer_preserves_system_and_user_roles_for_memory_continuation() -> None:
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

    rendered = AgentRunner._render_mallm_agent_prompt(
        agent,
        "general debate",
        "simple",
        continuation=True,
    )

    assert tokenizer.messages == [
        {"role": "system", "content": "general debate"},
        {"role": "user", "content": "simple"},
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


def test_agent_runner_persists_mallm_discussion_field_in_shared_memory() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        log_agents=False,
    )

    first_memory = runner._render_memory_entry(
        agent=runner.agents["A"],
        persona=("Fact Checker", "Checks factual plausibility."),
        response="Yes because the evidence supports it.",
        context="",
        question="Could this happen?",
        include_base=True,
    )
    later_memory = runner._render_memory_entry(
        agent=runner.agents["B"],
        persona=("Logic Reviewer", "Checks inference."),
        response="[AGREE]",
        context="",
        question="Could this happen?",
        include_base=False,
    )

    assert first_memory.startswith("This is the discussion to the current point:")
    assert "Fact Checker: Yes because the evidence supports it." in first_memory
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
        log_agents=False,
    )
    tokenizer = CapturingTokenizer()
    fake_agent = SimpleNamespace(model=SimpleNamespace(tokenizer=tokenizer))
    rendered = runner._render_memory_entry(
        agent=fake_agent,
        persona=("Fact Checker", "Checks facts."),
        response="Yes",
        context="",
        question="Question?",
        include_base=True,
    )

    assert [message["role"] for message in tokenizer.messages] == ["system", "user"]
    assert tokenizer.messages[0]["content"] == "This is the discussion to the current point: "
    assert tokenizer.messages[1]["content"] == "Fact Checker: Yes"
    assert rendered.startswith("<bos><system>")

def test_agent_runner_prompts_match_kv_native_mallm_memory_simple_structure() -> None:
    question = "Can penguins fly?"
    initial = AgentRunner._build_initial_user_content(
        "",
        question,
        persona=("Fact Checker", "Checks facts."),
    )
    ordinary = AgentRunner._build_followup_user_content(
        question,
        context="",
        persona=("Adversarial Reviewer", "Find missed evidence."),
        current_solution="No",
    )
    plain_initial = AgentRunner._build_plain_initial_prompt(
        "",
        question,
        persona=("Fact Checker", "Checks facts."),
    )

    assert initial.startswith("You take part in a discussion to solve a task.")
    assert "Task: Decide whether the answer to the following question is Yes or No." in initial
    assert "Input: Can penguins fly?" in initial
    assert "Your role: Fact Checker (Checks facts.)" in initial
    assert "Current Solution:" not in initial
    assert "This is the discussion to the current point:" not in initial

    assert ordinary.startswith("You take part in a discussion to solve a task.")
    assert "Your role: Adversarial Reviewer (Find missed evidence.)" in ordinary
    assert "Current Solution: No" in ordinary
    assert "This is the discussion to the current point:" not in ordinary
    assert (
        "Improve the current solution. If you agree with the current solution, answer with [AGREE], "
        "else answer with [DISAGREE] and explain why and provide an improved solution."
    ) in ordinary

    assert "Propose a solution." in plain_initial
    assert "Improve the current solution." not in plain_initial

def test_agent_runner_semantic_stance_prompt_hides_marker_and_final_label() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        log_agents=False,
    )
    runner.agent_personas["A"] = ("Fact Checker", "Checks facts.")

    class CapturingTokenizer:
        chat_template = "capture-template"

        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return "rendered-verification"

    tokenizer = CapturingTokenizer()
    fake_agent = SimpleNamespace(
        node_id="A",
        model=SimpleNamespace(tokenizer=tokenizer),
    )
    prompt = runner._build_reasoning_stance_prompt(
        agent=fake_agent,
        context="",
        question="Can penguins fly?",
        response="[DISAGREE] Penguins are flightless.\nFinal Solution: No",
    )

    assert prompt == "rendered-verification"
    assert [message["role"] for message in tokenizer.messages] == ["system", "user", "user"]
    assert "classify the stance" in tokenizer.messages[0]["content"]
    user_content = tokenizer.messages[1]["content"]
    assert "Task: Decide whether the answer to the following question is Yes or No." in user_content
    assert "Input: Can penguins fly?" in user_content
    assert "Penguins are flightless." in user_content
    assert "[DISAGREE]" not in user_content
    assert "Final Solution: No" not in user_content
    assert "Parsed control marker" not in user_content
    assert "Parsed claimed final answer" not in user_content
    assert "Current answer before this response" not in user_content
    assert "Fact Checker" not in "\n".join(message["content"] for message in tokenizer.messages)
    assert "SUPPORTS_YES" in tokenizer.messages[2]["content"]
    assert "SUPPORTS_NO" in tokenizer.messages[2]["content"]


def test_agent_runner_preserves_mallm_extracted_solution_as_current_draft() -> None:
    response = "[DISAGREE] The answer should be yes because the condition is satisfied."
    extracted_solution = "Yes"

    solution, answer, state = AgentRunner._response_updates_solution(
        response,
        current_solution="No",
        current_answer="No",
        extracted_solution=extracted_solution,
    )

    assert solution == extracted_solution
    assert answer == "Yes"
    assert state == "revise"
    assert response not in solution


def test_agent_runner_initial_semantic_mismatch_retries_before_commit() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        max_turns=2,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }

    calls = {"verifier": 0, "A": 0, "B": 0}

    def semantic_verifier(**kwargs):
        calls["verifier"] += 1
        # First pass rejects the internal consistency; retry pass accepts it.
        if calls["verifier"] == 1:
            return "INVALID", False
        return "VALID", True

    runner._verify_response_semantics_with_model = semantic_verifier

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls[agent.node_id] += 1
            text = "The evidence supports the proposition.\nFinal Solution: Yes" if agent.node_id == "A" else "[AGREE]"
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
            "edge_id": "edge",
            "offload_kind": "delta",
            "tokens_sent": 1,
            "tokens_received": 1,
            "expected_delta_tokens": 1,
        }
        return target_agent, meta, False, source_agent, meta

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    assert calls["A"] == 2
    assert calls["B"] == 1
    assert calls["verifier"] == 3
    assert result.agent_messages[0].solution == "Yes"
    assert result.agent_messages[0].final_answer == "Yes"
    assert result.agent_messages[0].verification_attempts == 1
    assert result.agent_messages[0].verification_syntax_failures == 0
    assert result.agent_messages[0].verification_semantic_failures == 1
    assert result.agent_messages[1].response_state == "agree"
    assert result.prediction == "Yes"
    assert result.accuracy == 1.0


def test_agent_runner_supermajority_consensus_uses_latest_vote_per_agent_after_full_participation() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=4,
        log_agents=False,
    )

    # The helper itself is a >66% supermajority over configured Agents. A missing
    # vote is an abstention; orchestration gates calls to this helper until every
    # configured Agent has completed its first verified Turn.
    assert runner._supermajority_consensus({"A": "Yes", "B": "Yes", "C": "Yes"}) == "Yes"

    # Supermajority is over unique Agents, not consecutive message history.
    assert runner._supermajority_consensus(
        {"A": "Yes", "B": "No", "C": "Yes", "D": "Yes"}
    ) == "Yes"
    assert runner._supermajority_consensus(
        {"A": "No", "B": "No", "C": "No", "D": "Yes"}
    ) == "No"
    assert runner._supermajority_consensus(
        {"A": "Yes", "B": "No", "C": "Yes", "D": "No"}
    ) is None

    # A later agent message replaces that Agent's old vote rather than adding a second vote.
    votes = {"A": "Yes", "B": "No", "C": "Yes", "D": "No"}
    votes["B"] = "Yes"
    assert runner._supermajority_consensus(votes) == "Yes"

    # The threshold is strictly greater than 0.66 (MALLM Supermajority uses 0.66).
    runner6 = AgentRunner(
        ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=6, log_agents=False
    )
    assert runner6._supermajority_consensus(
        {"A": "Yes", "B": "Yes", "C": "Yes", "D": "Yes", "E": "No", "F": "No"}
    ) == "Yes"  # 4/6 = 66.67% > 66%
    assert runner6._supermajority_consensus(
        {"A": "Yes", "B": "Yes", "C": "Yes", "D": "No", "E": "No", "F": "No"}
    ) is None

    runner8 = AgentRunner(
        ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=8, log_agents=False
    )
    assert runner8._supermajority_consensus(
        {"A": "Yes", "B": "Yes", "C": "Yes", "D": "Yes", "E": "Yes", "F": "No", "G": "No", "H": "No"}
    ) is None  # 5/8 = 62.5%
    assert runner8._supermajority_consensus(
        {"A": "Yes", "B": "Yes", "C": "Yes", "D": "Yes", "E": "Yes", "F": "Yes", "G": "No", "H": "No"}
    ) == "Yes"

    runner10 = AgentRunner(
        ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=10, log_agents=False
    )
    assert runner10._supermajority_consensus(
        {chr(ord("A") + i): ("Yes" if i < 6 else "No") for i in range(10)}
    ) is None
    assert runner10._supermajority_consensus(
        {chr(ord("A") + i): ("Yes" if i < 7 else "No") for i in range(10)}
    ) == "Yes"

    solution, answer, state = AgentRunner._response_updates_solution(
        "[AGREE]", current_solution="Yes", current_answer="Yes"
    )
    assert (solution, answer, state) == ("Yes", "Yes", "agree")

    solution, answer, state = AgentRunner._response_updates_solution(
        "[DISAGREE] The condition actually holds.",
        current_solution="No",
        current_answer="No",
        extracted_solution="Yes",
    )
    assert (solution, answer, state) == ("Yes", "Yes", "revise")


def test_agent_runner_final_majority_vote_is_used_when_supermajority_is_absent() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=5,
        log_agents=False,
    )
    votes = {"A": "Yes", "B": "Yes", "C": "Yes", "D": "No", "E": "No"}

    # 3/5 is not a >66% consensus, but it is the unique final majority.
    assert runner._supermajority_consensus(votes) is None
    answer, method = runner._majority_vote_with_random_tie(votes)
    assert answer == "Yes"
    assert method == "majority_vote"


def test_agent_runner_max_turns_uses_majority_vote_instead_of_latest_draft() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=5,
        max_turns=5,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._verify_response_semantics_with_model = lambda **kwargs: ("VALID", True)

    responses = {
        "A": "Final Solution: Yes",
        "B": "[DISAGREE] Additional support.\nFinal Solution: Yes",
        "C": "[DISAGREE] More support.\nFinal Solution: Yes",
        "D": "[DISAGREE] Counterargument.\nFinal Solution: No",
        "E": "[DISAGREE] Further counterargument.\nFinal Solution: No",
    }

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
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

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    assert result.turns[-1].vote_counts == {"Yes": 3, "No": 2, "Failed": 0}
    assert result.profile["consensus_reached"] is False
    assert result.agent_messages[-1].solution == "No"
    assert result.prediction == "Yes"
    assert result.profile["final_decision_method"] == "majority_vote"
    assert result.profile["final_decision_answer"] == "Yes"


def test_agent_runner_stops_early_when_supermajority_consensus_is_reached() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        max_turns=4,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._verify_response_semantics_with_model = lambda **kwargs: ("VALID", True)

    calls = []
    responses = {
        "A": "Final Solution: Yes",
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

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    assert calls == ["A", "B"]
    assert len(result.agent_messages) == 2
    assert result.prediction == "Yes"
    assert result.profile["consensus_reached"] is True
    assert result.profile["consensus_turn"] == 2
    assert result.profile["persona_generator"] == "expert"
    assert result.profile["response_generator"] == "simple"
    assert result.profile["discussion_paradigm"] == "memory"
    assert result.profile["decision_protocol"] == "turn_supermajority_then_majority_vote"
    assert result.profile["requested_max_turns"] == 4
    assert result.profile["num_agent_messages"] == 2
    assert result.profile["completed_turns"] == 2
    assert len(result.turns) == 2
    assert result.turns[0].consensus_reached is False
    assert result.turns[1].consensus_answer == "Yes"


def test_agent_runner_defers_supermajority_until_every_agent_participates_once() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=4,
        max_turns=4,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._verify_response_semantics_with_model = lambda **kwargs: ("VALID", True)

    responses = {
        "A": "Final Solution: Yes",
        "B": "[DISAGREE] The rationale needs correction.\nFinal Solution: Yes",
        "C": "[AGREE]\nFinal Solution: Yes",
        "D": "[DISAGREE]\nFinal Solution: No",
    }
    calls = []

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

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    # A/B/C already form 3/4 = 75%, but consensus must not be evaluated before
    # D completes the mandatory first participation pass. D speaks on Turn 4; only
    # then is 3/4 > 66% accepted as the first eligible consensus.
    assert calls == ["A", "B", "C", "D"]
    assert len(result.turns) == 4
    assert result.turns[2].vote_counts == {"Yes": 3, "No": 0, "Failed": 0}
    assert result.turns[2].consensus_reached is False
    assert result.turns[3].vote_counts == {"Yes": 3, "No": 1, "Failed": 0}
    assert result.turns[3].consensus_answer == "Yes"
    assert result.profile["consensus_reached"] is True
    assert result.profile["consensus_turn"] == 4
    assert result.profile["consensus_requires_full_initial_participation"] is True
    assert result.prediction == "Yes"


def test_agent_runner_max_turns_counts_individual_agent_turns() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=4,
        max_turns=8,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._verify_response_semantics_with_model = lambda **kwargs: ("VALID", True)
    calls = []
    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls.append(agent.node_id)
            if "Current Solution:" not in prompt_text:
                text = "Final Solution: Yes"
            elif agent.node_id in {"A", "C"}:
                text = "[DISAGREE]\nFinal Solution: Yes"
            else:
                text = "[DISAGREE]\nFinal Solution: No"
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

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["No"])

    assert calls == ["A", "B", "C", "D", "A", "B", "C", "D"]
    assert len(result.agent_messages) == 8
    assert result.profile["consensus_reached"] is False
    assert result.profile["num_agent_messages"] == 8
    assert result.profile["completed_turns"] == 8
    assert len(result.turns) == 8
    assert [turn.turn_index for turn in result.turns] == list(range(1, 9))

def test_agent_runner_consensus_is_evaluated_after_every_turn() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=4,
        max_turns=8,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._verify_response_semantics_with_model = lambda **kwargs: ("VALID", True)

    calls = []
    # The first four Turns leave a 2:2 split. On Turn 5, A changes from Yes
    # to No, producing 3:1 for No. Turn-based consensus must stop immediately,
    # before B/C/D take another turn.
    scripted = [
        ("A", "Final Solution: Yes"),
        ("B", "[DISAGREE]\nFinal Solution: No"),
        ("C", "[DISAGREE]\nFinal Solution: Yes"),
        ("D", "[DISAGREE]\nFinal Solution: No"),
        ("A", "[DISAGREE]\nFinal Solution: No"),
    ]
    cursor = {"i": 0}

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            expected_agent, text = scripted[cursor["i"]]
            assert agent.node_id == expected_agent
            cursor["i"] += 1
            calls.append(agent.node_id)
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

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["No"])

    assert calls == ["A", "B", "C", "D", "A"]
    assert len(result.turns) == 5
    assert result.turns[3].vote_counts == {"Yes": 2, "No": 2, "Failed": 0}
    assert result.turns[3].consensus_reached is False
    assert result.turns[4].vote_counts == {"Yes": 1, "No": 3, "Failed": 0}
    assert result.turns[4].consensus_answer == "No"
    assert result.profile["consensus_reached"] is True
    assert result.profile["consensus_turn"] == 5


def test_agent_runner_semantic_verifier_parsers_are_strict() -> None:
    assert AgentRunner._parse_reasoning_stance("SUPPORTS_YES") == "Yes"
    assert AgentRunner._parse_reasoning_stance("SUPPORTS_NO") == "No"
    assert AgentRunner._parse_reasoning_stance("UNCLEAR") == "unclear"
    assert AgentRunner._parse_reasoning_stance("CONTRADICTORY") == "contradictory"
    assert AgentRunner._parse_reasoning_stance("SUPPORTS_YES because ...") is None
    assert AgentRunner._parse_reasoning_stance("Yes") is None
    assert AgentRunner._parse_improvement_verdict("SUBSTANTIVE") is True
    assert AgentRunner._parse_improvement_verdict("NOT_SUBSTANTIVE") is False
    assert AgentRunner._parse_improvement_verdict("SUBSTANTIVE because ...") is None


def test_agent_runner_semantic_verifier_compares_independent_reasoning_stance_to_final_answer() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        log_agents=False,
    )
    fake_agent = SimpleNamespace(node_id="A", model=SimpleNamespace(tokenizer=SimpleNamespace()))
    runner._build_reasoning_stance_prompt = lambda **kwargs: "stance-prompt"
    seen = []

    def fake_run(*, agent, prompt):
        seen.append(prompt)
        return "SUPPORTS_YES"

    runner._run_semantic_verifier_prompt = fake_run
    raw, passed = runner._verify_response_semantics_with_model(
        agent=fake_agent,
        context="",
        question="Question?",
        response="[DISAGREE] The reasoning supports Yes.\nFinal Solution: No",
        marker="disagree",
        final_answer="No",
        current_answer="Yes",
        current_response="[AGREE] Previous reasoning.\nFinal Solution: Yes",
    )
    assert seen == ["stance-prompt"]
    assert raw == "stance=Yes; final=No"
    assert passed is False


def test_agent_runner_same_answer_disagree_requires_substantive_reasoning_change() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        log_agents=False,
    )
    fake_agent = SimpleNamespace(node_id="A", model=SimpleNamespace(tokenizer=SimpleNamespace()))
    runner._build_reasoning_stance_prompt = lambda **kwargs: "stance-prompt"
    runner._build_reasoning_improvement_prompt = lambda **kwargs: "improvement-prompt"
    outputs = iter(["SUPPORTS_NO", "NOT_SUBSTANTIVE"])
    seen = []

    def fake_run(*, agent, prompt):
        seen.append(prompt)
        return next(outputs)

    runner._run_semantic_verifier_prompt = fake_run
    raw, passed = runner._verify_response_semantics_with_model(
        agent=fake_agent,
        context="",
        question="Question?",
        response="[DISAGREE] The same point in different words.\nFinal Solution: No",
        marker="disagree",
        final_answer="No",
        current_answer="No",
        current_response="[AGREE] The original point.\nFinal Solution: No",
    )
    assert seen == ["stance-prompt", "improvement-prompt"]
    assert raw == "stance=No; improvement=NOT_SUBSTANTIVE"
    assert passed is False


def test_agent_runner_semantic_reasoning_body_removes_control_and_final_syntax() -> None:
    response = (
        "[DISAGREE]\nThe prior rationale is wrong because the evidence points the other way.\n"
        "Final Solution: Yes"
    )
    assert AgentRunner._semantic_reasoning_body(response) == (
        "The prior rationale is wrong because the evidence points the other way."
    )


def test_agent_runner_agreement_marker_requires_bracketed_first_token() -> None:
    assert AgentRunner._extract_agreement_marker("[AGREE]\nFinal Solution: Yes") == "agree"
    assert AgentRunner._extract_agreement_marker("  [DISAGREE] because...") == "disagree"
    assert AgentRunner._extract_agreement_marker("I DISAGREE. Final Solution: No") is None
    assert AgentRunner._extract_agreement_marker("Reasoning says I agree, but Final Solution: Yes") is None


def test_agent_runner_followup_syntax_is_checked_before_semantics() -> None:
    missing_marker = AgentRunner._verify_followup_syntax("Final Solution: Yes")
    assert missing_marker.passed is False
    assert missing_marker.syntax_passed is False
    assert missing_marker.semantic_passed is None
    assert "[AGREE]/[DISAGREE]" in missing_marker.syntax_reason

    missing_revision = AgentRunner._verify_followup_syntax("[DISAGREE] The wording is wrong.")
    assert missing_revision.syntax_passed is False
    assert missing_revision.semantic_passed is None
    assert "Final Solution" in missing_revision.syntax_reason

    valid = AgentRunner._verify_followup_syntax(
        "[DISAGREE] The reasoning supports the opposite conclusion.\nFinal Solution: Yes"
    )
    assert valid.syntax_passed is True
    assert valid.marker == "disagree"
    assert valid.final_answer == "Yes"

    bare_agree = AgentRunner._verify_followup_syntax("[AGREE]")
    assert bare_agree.syntax_passed is True
    assert bare_agree.marker == "agree"
    assert bare_agree.final_answer is None

    # Stage 1 must never infer a vote from arbitrary prose. Expanded follow-ups
    # require the exact terminal Final Solution line.
    inferred_only = AgentRunner._verify_followup_syntax(
        "[AGREE] I still think the answer is Yes because the reasoning is sound."
    )
    assert inferred_only.syntax_passed is False
    assert inferred_only.semantic_passed is None
    assert inferred_only.final_answer is None

    initial_inferred_only = AgentRunner._verify_initial_syntax(
        "The reasoning points to Yes, so the answer is Yes."
    )
    assert initial_inferred_only.syntax_passed is False
    assert initial_inferred_only.semantic_passed is None
    assert initial_inferred_only.final_answer is None


def test_agent_runner_final_solution_yes_is_committed_when_semantically_valid() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        max_turns=2,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._verify_response_semantics_with_model = lambda **kwargs: ("VALID", True)

    responses = {
        "A": "The initial reasoning supports No.\nFinal Solution: No",
        "B": (
            "[DISAGREE] Some trees live for thousands of years, longer than the roughly "
            "two-thousand-year Common Era.\nFinal Solution: Yes"
        ),
    }

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
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

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    b = result.agent_messages[1]
    assert b.verification_syntax_passed is True
    assert b.verification_semantic_passed is True
    assert b.verification_passed is True
    assert b.final_answer == "Yes"
    assert b.response_state == "revise"
    assert b.solution == "Yes"
    assert result.turns[-1].vote_counts == {"Yes": 1, "No": 1, "Failed": 0}
    assert result.profile["final_decision_method"] == "random_tie_break"


def test_agent_runner_initial_proposal_retries_until_yes_no_is_explicit() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        max_turns=2,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._verify_response_semantics_with_model = lambda **kwargs: ("VALID", True)

    calls = {"A": 0, "B": 0}

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls[agent.node_id] += 1
            if agent.node_id == "A" and calls["A"] <= 5:
                text = "I cannot determine a final answer from the available information."
            elif agent.node_id == "A":
                text = "The evidence supports the proposition.\nFinal Solution: Yes"
            else:
                text = "[AGREE]"
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
            "edge_id": "edge",
            "offload_kind": "delta",
            "tokens_sent": 1,
            "tokens_received": 1,
            "expected_delta_tokens": 1,
        }
        return target_agent, meta, False, source_agent, meta

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    assert calls == {"A": 6, "B": 1}
    first = result.agent_messages[0]
    assert first.response.endswith("Final Solution: Yes")
    assert first.final_answer == "Yes"
    assert first.verification_attempts == 5
    assert first.verification_passed is True
    assert first.verification_reason == "ok"
    assert result.prediction == "Yes"
    assert result.profile["verification_retry_policy"] == "max_10_then_agent_failure"
    assert result.profile["verification_retry_count"] == 5


def test_agent_runner_verification_retry_accepts_corrected_response() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        max_turns=2,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._verify_response_semantics_with_model = lambda **kwargs: ("VALID", True)

    calls = {"A": 0, "B": 0}

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls[agent.node_id] += 1
            if agent.node_id == "A":
                text = "Final Solution: Yes"
            elif calls["B"] <= 5:
                text = "[AGREE]\nFinal Solution: No"
            else:
                text = "[AGREE]\nFinal Solution: Yes"
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
            "edge_id": "edge",
            "offload_kind": "delta",
            "tokens_sent": 1,
            "tokens_received": 1,
            "expected_delta_tokens": 1,
        }
        return target_agent, meta, False, source_agent, meta

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    assert calls == {"A": 1, "B": 6}
    assert len(result.agent_messages) == 2
    corrected = result.agent_messages[1]
    assert corrected.response == "[AGREE]\nFinal Solution: Yes"
    assert corrected.verification_attempts == 5
    assert corrected.verification_passed is True
    assert corrected.agreement_marker == "agree"
    assert corrected.final_answer == "Yes"
    assert corrected.response_state == "agree"
    assert result.profile["consensus_reached"] is True
    assert result.profile["verification_retry_policy"] == "max_10_then_agent_failure"
    assert result.profile["verification_retry_count"] == 5
    assert result.profile["verification_failure_count"] == 0


def test_agent_runner_uses_random_tie_break_instead_of_last_draft_when_max_turns_end(monkeypatch) -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=4,
        max_turns=4,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._verify_response_semantics_with_model = lambda **kwargs: ("VALID", True)

    calls = []

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls.append(agent.node_id)
            if "Current Solution:" not in prompt_text:
                text = "Final Solution: Yes"
            elif prompt_text.split("Current Solution:", 1)[1].lstrip().startswith("Yes"):
                text = "[DISAGREE]\nFinal Solution: No"
            else:
                text = "[DISAGREE]\nFinal Solution: Yes"
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

    runner._star_offload_to_agent = fake_star_offload
    monkeypatch.setattr("core.agent_runner.random.choice", lambda winners: "Yes")
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    assert calls == ["A", "B", "C", "D"]
    assert len(result.agent_messages) == 4
    # The latest draft is No, but a tied final vote must use random tie-breaking
    # instead of falling back to that latest draft.
    assert result.agent_messages[-1].solution == "No"
    assert result.prediction == "Yes"
    assert result.profile["consensus_reached"] is False
    assert result.profile["consensus_turn"] is None
    assert result.profile["final_decision_method"] == "random_tie_break"
    assert result.profile["final_decision_answer"] == "Yes"



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


def test_agent_runner_zero_delta_handoff_after_verification_rollback_is_noop() -> None:
    """A rejected retry can leave source and next target on the same shared prefix."""
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        cache_mode="free",
        log_agents=False,
    )
    source = runner.agents["B"]
    target = runner.agents["A"]
    source.append_context_text("shared-prefix")
    target.append_context_text("shared-prefix")
    target_tokens = target.cache_seq_len

    prefix_tokens, delta_tokens, matched = runner._build_missing_cache_delta(
        source_agent=source,
        target_agent=target,
    )
    assert prefix_tokens == target_tokens
    assert delta_tokens == 0
    assert matched is True

    metadata, cleared = runner._offload_delta_hop(
        source_agent=source,
        target_agent=target,
    )

    assert metadata["offload_kind"] == "noop"
    assert metadata["tokens_sent"] == 0
    assert metadata["tokens_received"] == 0
    assert metadata["expected_delta_tokens"] == 0
    assert metadata["target_tokens_before_replay"] == target_tokens
    assert metadata["target_tokens_after_replay"] == target_tokens
    assert target.cache_seq_len == target_tokens
    assert cleared is True
    assert source.past_key_values is None
    assert source.cache_token_ids == []


def test_agent_runner_prepares_second_hop_when_first_hop_is_zero_delta(monkeypatch) -> None:
    """B->A may be a no-op after rollback while A->C still needs preparation."""
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=3,
        cache_mode="retain",
        log_agents=False,
    )
    hub = runner.hub_agent
    source = runner.agents["B"]
    target = runner.agents["C"]
    hub.append_context_text("shared-prefix")
    source.append_context_text("shared-prefix")

    def fail_refresh(**kwargs):
        raise AssertionError("zero-delta B->hub must not invoke translation")

    seen = {}

    def fake_build(*, source_agent, target_agent, source_past_key_values, source_token_ids, **kwargs):
        seen["source"] = source_agent.node_id
        seen["target"] = target_agent.node_id
        seen["tokens"] = list(source_token_ids)
        return "A_to_B", source_past_key_values, list(source_token_ids)

    monkeypatch.setattr(runner.cache_translator, "refresh_pretranslated_cache", fail_refresh)
    monkeypatch.setattr(runner.cache_translator, "build_pretranslated_past_for_edge", fake_build)

    runner._prepare_outgoing_route_translation(
        source_agent=source,
        logical_target_agent=target,
    )

    assert seen["source"] == hub.node_id
    assert seen["target"] == target.node_id
    assert seen["tokens"] == source.cache_token_ids == hub.cache_token_ids
    pending = runner._pending_pretranslated_second_hops[(source.node_id, target.node_id)]
    assert pending[0] == "A_to_B"
    assert pending[2] == source.cache_token_ids


def _fake_past(seq_len: int):
    key = torch.zeros((1, 1, seq_len, 1), dtype=torch.float32)
    value = torch.zeros((1, 1, seq_len, 1), dtype=torch.float32)
    return ((key, value),)


def test_live_kv_cache_bytes_and_peak_track_actual_cache_storage() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        log_agents=False,
    )
    runner.agents["A"].past_key_values = _fake_past(12)
    runner._update_peak_memory_breakdown()
    assert runner._live_kv_cache_bytes() == 12 * 8
    assert runner._peak_kv_cache_bytes == 12 * 8

    runner.agents["B"].past_key_values = _fake_past(16)
    runner.agents["A"].pretranslated_past_by_edge["A_to_B"] = _fake_past(12)
    runner._update_peak_memory_breakdown()
    assert runner._live_kv_cache_bytes() == 40 * 8
    assert runner._peak_kv_cache_bytes == 40 * 8


def test_agent_runner_syntax_failure_retries_before_semantic_verifier() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        max_turns=2,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }

    verifier_calls = {"count": 0}

    def semantic_verifier(**kwargs):
        verifier_calls["count"] += 1
        return "VALID", True

    runner._verify_response_semantics_with_model = semantic_verifier
    calls = {"A": 0, "B": 0}

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls[agent.node_id] += 1
            if agent.node_id == "A":
                text = "Initial reasoning.\nFinal Solution: Yes"
            elif calls["B"] == 1:
                # Syntactically invalid: no first-token control marker.
                text = "I agree with the current solution.\nFinal Solution: Yes"
            else:
                text = "[AGREE]\nThe reasoning remains consistent.\nFinal Solution: Yes"
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
            "edge_id": "edge",
            "offload_kind": "delta",
            "tokens_sent": 1,
            "tokens_received": 1,
            "expected_delta_tokens": 1,
        }
        return target_agent, meta, False, source_agent, meta

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    # Semantic verification ran for A's valid initial proposal and only B's
    # second, syntactically valid attempt. B's malformed first attempt never
    # reached the semantic verifier.
    assert verifier_calls["count"] == 2
    assert calls == {"A": 1, "B": 2}
    b = result.agent_messages[1]
    assert b.verification_attempts == 1
    assert b.verification_syntax_failures == 1
    assert b.verification_semantic_failures == 0
    assert b.verification_syntax_passed is True
    assert b.verification_semantic_passed is True
    assert b.verification_passed is True


def test_agent_ttft_measurement_does_not_change_generation_result() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    model = ctx.tp.get_model("A")

    baseline = Agent(
        node_id="A",
        model=model,
        device="cpu",
        max_new_tokens=3,
        temperature=0.0,
    )
    measured = Agent(
        node_id="A",
        model=model,
        device="cpu",
        max_new_tokens=3,
        temperature=0.0,
    )
    measured.measure_first_token_ttft = True

    baseline_generation = baseline.generate_response("abc")
    measured_generation = measured.generate_response("abc")

    assert measured_generation.generated_token_ids == baseline_generation.generated_token_ids
    assert measured_generation.text == baseline_generation.text
    assert baseline_generation.first_token_ttft_sec is None
    assert measured_generation.first_token_ttft_sec is not None
    assert measured_generation.first_token_ttft_sec >= 0.0


def test_agent_runner_ttft_includes_route_cost_and_excludes_verification_retries() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        max_turns=2,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._verify_response_semantics_with_model = lambda **kwargs: ("VALID", True)
    runner._update_peak_memory_breakdown = lambda: None
    runner._prepare_outgoing_route_translation = lambda **kwargs: None

    phase_times = iter([0.2, 0.3])  # next-turn pretranslation, then offload/replay

    def fake_measure_phase(fn):
        result = fn()
        return result, next(phase_times)

    runner._measure_ttft_phase = fake_measure_phase

    calls = {"A": 0, "B": 0}
    measurement_flags = []

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls[agent.node_id] += 1
            measurement_flags.append((agent.node_id, calls[agent.node_id], agent.measure_first_token_ttft))
            if agent.node_id == "A":
                text = "Final Solution: Yes"
                first_token_ttft = 0.1
            elif calls["B"] == 1:
                # Force one corrective verification retry. Its first-token latency
                # belongs to the Turn; the retry below must not be added again.
                text = "[AGREE]\nFinal Solution: No"
                first_token_ttft = 0.4
            else:
                text = "[AGREE]\nFinal Solution: Yes"
                first_token_ttft = 99.0
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
                first_token_ttft_sec=first_token_ttft,
            )
        return generate

    for agent in runner.agent_sequence:
        agent.generate_response = fake_generate(agent)

    def fake_star_offload(*, source_agent, target_agent):
        meta = {
            "edge_id": "edge",
            "offload_kind": "delta",
            "tokens_sent": 1,
            "tokens_received": 1,
            "expected_delta_tokens": 1,
        }
        return target_agent, meta, False, source_agent, meta

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    assert calls == {"A": 1, "B": 2}
    assert measurement_flags == [
        ("A", 1, True),
        ("B", 1, True),
        ("B", 2, False),
    ]
    assert result.agent_messages[0].ttft_sec == pytest.approx(0.1)
    assert result.agent_messages[1].ttft_sec == pytest.approx(0.2 + 0.3 + 0.4)
    assert result.profile["turn_ttft_sec"] == pytest.approx([0.1, 0.9])
    assert result.profile["example_ttft_sec"] == pytest.approx(0.5)
    assert result.profile["ttft_includes_offload_translation"] is True
    assert result.profile["ttft_excludes_verification_retries"] is True


def _install_fake_cache(agent, token_ids):
    token_ids = list(token_ids)
    agent.past_key_values = _fake_past(len(token_ids))
    agent.cache_token_ids = token_ids
    agent.invalidate_pretranslated_caches()


def test_retain_direct_handoff_translates_only_missing_delta(monkeypatch) -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        cache_mode="retain",
        log_agents=False,
    )
    source = runner.hub_agent
    target = runner.agents["B"]
    source_ids = list(range(10))
    _install_fake_cache(source, source_ids)
    _install_fake_cache(target, source_ids[:6])

    seen = {}

    def fake_build(*, source_agent, target_agent, source_past_key_values, source_token_ids, **kwargs):
        seen["source"] = source_agent.node_id
        seen["target"] = target_agent.node_id
        seen["past_tokens"] = get_past_seq_len(source_past_key_values)
        seen["token_ids"] = list(source_token_ids)
        seen["kwargs"] = dict(kwargs)
        edge = runner.cache_translator._get_edge(source_agent.node_id, target_agent.node_id)
        return edge.id, source_past_key_values, list(source_token_ids)

    monkeypatch.setattr(runner.cache_translator, "build_pretranslated_past_for_edge", fake_build)

    runner._prepare_outgoing_route_translation(source_agent=source, logical_target_agent=target)

    assert seen["past_tokens"] == 4
    assert seen["token_ids"] == source_ids[6:]
    assert list(seen["kwargs"]["retain_source_full_token_ids"]) == source_ids
    assert seen["kwargs"]["retain_source_full_past_key_values"] is source.past_key_values
    assert seen["kwargs"]["retain_target_prefix_past_key_values"] is target.past_key_values
    metadata, cleared = runner._offload_delta_hop(source_agent=source, target_agent=target)
    assert metadata["tokens_sent"] == 4
    assert target.cache_token_ids == source_ids
    assert target.cache_seq_len == 10
    assert cleared is False


def test_free_direct_handoff_preserves_full_translation_then_delta_offload(monkeypatch) -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        cache_mode="free",
        log_agents=False,
    )
    source = runner.hub_agent
    target = runner.agents["B"]
    source_ids = list(range(10))
    _install_fake_cache(source, source_ids)
    _install_fake_cache(target, source_ids[:6])

    seen = {}

    def fake_build(*, source_agent, target_agent, source_past_key_values, source_token_ids, **kwargs):
        seen["past_tokens"] = get_past_seq_len(source_past_key_values)
        seen["token_ids"] = list(source_token_ids)
        seen["kwargs"] = dict(kwargs)
        edge = runner.cache_translator._get_edge(source_agent.node_id, target_agent.node_id)
        return edge.id, source_past_key_values, list(source_token_ids)

    monkeypatch.setattr(runner.cache_translator, "build_pretranslated_past_for_edge", fake_build)

    runner._prepare_outgoing_route_translation(source_agent=source, logical_target_agent=target)

    assert seen["past_tokens"] == 10
    assert seen["token_ids"] == source_ids
    assert seen["kwargs"] == {}
    metadata, cleared = runner._offload_delta_hop(source_agent=source, target_agent=target)
    assert metadata["tokens_sent"] == 4
    assert target.cache_token_ids == source_ids
    assert target.cache_seq_len == 10
    assert cleared is False  # hub is intentionally retained in free mode


def test_free_nonhub_to_hub_pretranslation_uses_resident_hub_prefix(monkeypatch) -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        cache_mode="free",
        log_agents=False,
    )
    hub = runner.hub_agent
    source = runner.agents["B"]
    source_ids = list(range(10))
    _install_fake_cache(source, source_ids)
    _install_fake_cache(hub, source_ids[:6])

    seen = {}

    def fake_build(*, source_agent, target_agent, source_past_key_values, source_token_ids, **kwargs):
        seen["past_tokens"] = get_past_seq_len(source_past_key_values)
        seen["token_ids"] = list(source_token_ids)
        seen["kwargs"] = dict(kwargs)
        edge = runner.cache_translator._get_edge(source_agent.node_id, target_agent.node_id)
        return edge.id, source_past_key_values, list(source_token_ids)

    monkeypatch.setattr(runner.cache_translator, "build_pretranslated_past_for_edge", fake_build)

    runner._prepare_outgoing_route_translation(source_agent=source, logical_target_agent=hub)

    assert seen["past_tokens"] == 4
    assert seen["token_ids"] == source_ids[6:]
    assert list(seen["kwargs"]["retain_source_full_token_ids"]) == source_ids
    assert seen["kwargs"]["retain_source_full_past_key_values"] is source.past_key_values
    assert seen["kwargs"]["retain_target_prefix_past_key_values"] is hub.past_key_values

    metadata, cleared = runner._offload_delta_hop(source_agent=source, target_agent=hub)
    assert metadata["tokens_sent"] == 4
    assert hub.cache_token_ids == source_ids
    assert hub.cache_seq_len == 10
    assert cleared is True


def test_free_and_retain_prepare_identical_nonhub_to_hub_mot_delta(monkeypatch) -> None:
    prepared = {}

    for mode in ("free", "retain"):
        ctx = _ctx("tiny-a,tiny-a")
        runner = AgentRunner(
            ctx=ctx,
            translator_pool=ctx.tp,
            alg="mot",
            agent_count=3,
            cache_mode=mode,
            log_agents=False,
        )
        hub = runner.hub_agent
        source = runner.agents["B"]
        target = runner.agents["C"]
        source_ids = list(range(10))
        _install_fake_cache(source, source_ids)
        _install_fake_cache(hub, source_ids[:6])
        # Free normally has an empty upcoming non-hub target; Retain may keep one.
        # Keeping C empty in both isolates the physical B->hub hop that must be
        # mathematically identical across cache modes.
        target.clear_kv_cache()

        seen = []

        def fake_build(*, source_agent, target_agent, source_past_key_values, source_token_ids, **kwargs):
            seen.append({
                "src": source_agent.node_id,
                "dst": target_agent.node_id,
                "past_tokens": get_past_seq_len(source_past_key_values),
                "token_ids": list(source_token_ids),
                "full_ids": list(kwargs.get("retain_source_full_token_ids", [])),
                "full_past_tokens": (
                    get_past_seq_len(kwargs["retain_source_full_past_key_values"])
                    if kwargs.get("retain_source_full_past_key_values") is not None else 0
                ),
                "target_prefix_tokens": (
                    get_past_seq_len(kwargs["retain_target_prefix_past_key_values"])
                    if kwargs.get("retain_target_prefix_past_key_values") is not None else 0
                ),
            })
            edge = runner.cache_translator._get_edge(source_agent.node_id, target_agent.node_id)
            return edge.id, source_past_key_values, list(source_token_ids)

        monkeypatch.setattr(runner.cache_translator, "build_pretranslated_past_for_edge", fake_build)
        runner._prepare_outgoing_route_translation(source_agent=source, logical_target_agent=target)
        prepared[mode] = seen[0]

    assert prepared["free"] == prepared["retain"] == {
        "src": "B",
        "dst": "A",
        "past_tokens": 4,
        "token_ids": [6, 7, 8, 9],
        "full_ids": list(range(10)),
        "full_past_tokens": 10,
        "target_prefix_tokens": 6,
    }


def test_retain_two_hop_route_translates_each_physical_delta_only(monkeypatch) -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=3,
        cache_mode="retain",
        log_agents=False,
    )
    hub = runner.hub_agent
    source = runner.agents["B"]
    target = runner.agents["C"]
    source_ids = list(range(10))
    _install_fake_cache(source, source_ids)
    _install_fake_cache(hub, source_ids[:6])
    _install_fake_cache(target, source_ids[:4])

    seen = []

    def fake_build(*, source_agent, target_agent, source_past_key_values, source_token_ids, **kwargs):
        seen.append(
            (
                source_agent.node_id,
                target_agent.node_id,
                get_past_seq_len(source_past_key_values),
                list(source_token_ids),
                dict(kwargs),
            )
        )
        edge = runner.cache_translator._get_edge(source_agent.node_id, target_agent.node_id)
        return edge.id, source_past_key_values, list(source_token_ids)

    monkeypatch.setattr(runner.cache_translator, "build_pretranslated_past_for_edge", fake_build)

    runner._prepare_outgoing_route_translation(source_agent=source, logical_target_agent=target)

    # B->hub translates only 6..9; future hub->C translates only 4..9.
    assert seen[0][2:4] == (4, source_ids[6:])
    assert seen[1][2:4] == (6, source_ids[4:])
    assert list(seen[0][4]["retain_source_full_token_ids"]) == source_ids
    assert seen[0][4]["retain_source_full_past_key_values"] is source.past_key_values
    assert seen[0][4]["retain_target_prefix_past_key_values"] is hub.past_key_values
    assert list(seen[1][4]["retain_source_full_token_ids"]) == source_ids
    assert get_past_seq_len(seen[1][4]["retain_source_full_past_key_values"]) == len(source_ids)
    assert seen[1][4]["retain_target_prefix_past_key_values"] is target.past_key_values

    _, first_meta, source_cleared, _, second_meta = runner._star_offload_to_agent(
        source_agent=source,
        target_agent=target,
    )
    assert first_meta["tokens_sent"] == 4
    assert second_meta["tokens_sent"] == 6
    assert hub.cache_token_ids == source_ids
    assert target.cache_token_ids == source_ids
    assert hub.cache_seq_len == 10
    assert target.cache_seq_len == 10
    assert source_cleared is False


def test_free_two_hop_route_conditions_first_delta_on_resident_hub_prefix(monkeypatch) -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=3,
        cache_mode="free",
        log_agents=False,
    )
    hub = runner.hub_agent
    source = runner.agents["B"]
    target = runner.agents["C"]
    source_ids = list(range(10))
    _install_fake_cache(source, source_ids)
    _install_fake_cache(hub, source_ids[:6])
    _install_fake_cache(target, source_ids[:4])

    seen = []

    def fake_build(*, source_agent, target_agent, source_past_key_values, source_token_ids, **kwargs):
        seen.append(
            (
                source_agent.node_id,
                target_agent.node_id,
                get_past_seq_len(source_past_key_values),
                list(source_token_ids),
                dict(kwargs),
            )
        )
        edge = runner.cache_translator._get_edge(source_agent.node_id, target_agent.node_id)
        return edge.id, source_past_key_values, list(source_token_ids)

    monkeypatch.setattr(runner.cache_translator, "build_pretranslated_past_for_edge", fake_build)

    runner._prepare_outgoing_route_translation(source_agent=source, logical_target_agent=target)

    # The first physical hop enters the resident hub, so Free must compute the
    # missing suffix against the exact hub prefix just like Retain.  The second
    # hop still sees the full future hub because a Free non-hub target is rebuilt.
    assert seen[0][2:4] == (4, source_ids[6:])
    assert list(seen[0][4]["retain_source_full_token_ids"]) == source_ids
    assert seen[0][4]["retain_source_full_past_key_values"] is source.past_key_values
    assert seen[0][4]["retain_target_prefix_past_key_values"] is hub.past_key_values
    assert seen[1][2:4] == (10, source_ids)
    assert seen[1][4] == {}

    _, first_meta, source_cleared, _, second_meta = runner._star_offload_to_agent(
        source_agent=source,
        target_agent=target,
    )
    assert first_meta["tokens_sent"] == 4
    assert second_meta["tokens_sent"] == 6
    assert hub.cache_token_ids == source_ids
    assert target.cache_token_ids == source_ids
    assert target.cache_seq_len == 10
    assert source_cleared is True


def test_heterogeneous_shared_memory_is_model_neutral_across_chat_templates() -> None:
    class ExplodingChatTokenizer:
        def __init__(self, marker: str) -> None:
            self.chat_template = f"template-{marker}"
            self.marker = marker
            self.calls = 0
        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            return f"<{self.marker}>{messages[-1]['content']}</{self.marker}>"

    ctx = Context(SimpleNamespace(model_ids="tiny-a,tiny-b", model_directions="all", device="cpu", dtype="float32"))
    runner = AgentRunner(ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=2, log_agents=False)
    llama_like = ExplodingChatTokenizer("llama-control")
    qwen_like = ExplodingChatTokenizer("qwen-control")
    fake_a = SimpleNamespace(model=SimpleNamespace(tokenizer=llama_like))
    fake_b = SimpleNamespace(model=SimpleNamespace(tokenizer=qwen_like))
    first = runner._render_memory_entry(
        agent=fake_a, persona=("Historian", "Checks chronology."), response="Final Solution: No",
        context="", question="Question?", include_base=True,
    )
    later = runner._render_memory_entry(
        agent=fake_b, persona=("Reviewer", "Checks reasoning."),
        response="[DISAGREE] Correction. Final Solution: Yes", context="", question="Question?", include_base=False,
    )
    assert first == "This is the discussion to the current point: \nHistorian: Final Solution: No\n"
    assert later == "\nReviewer: [DISAGREE] Correction. Final Solution: Yes\n"
    assert llama_like.calls == 0 and qwen_like.calls == 0


def test_heterogeneous_delta_uses_target_token_grid(monkeypatch) -> None:
    from core.common import TokenIDs
    ctx = Context(SimpleNamespace(model_ids="tiny-a,tiny-b", model_directions="all", device="cpu", dtype="float32"))
    runner = AgentRunner(ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=2, log_agents=False)
    source, target = runner.agent_sequence
    source.past_key_values = _fake_past(3); source.cache_token_ids = [1,2,3]
    target.past_key_values = _fake_past(1); target.cache_token_ids = [91]
    def fake_retokenize(*, source_model, target_model, source_context_token_ids):
        return TokenIDs(torch.tensor([[91,92]], dtype=torch.long), model_id=target_model.id)
    monkeypatch.setattr("alg.mot.train.retokenize_agent_runner_context", fake_retokenize)
    assert runner._build_missing_cache_delta(source_agent=source, target_agent=target) == (1,1,True)


def test_heterogeneous_retain_cross_source_appends_to_hub_without_rewriting_same_model_prefix(monkeypatch) -> None:
    from core.common import TokenIDs

    ctx = Context(SimpleNamespace(model_ids="tiny-a,tiny-b", model_directions="all", device="cpu", dtype="float32"))
    runner = AgentRunner(ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=4, cache_mode="retain", log_agents=False)
    hub, source, target = runner.hub_agent, runner.agents["B"], runner.agents["C"]

    # Canonical Hub/C history is [77]. B received that logical history on its own
    # token grid [1,2], then committed exactly one new Memory token [3].
    hub.past_key_values = _fake_past(1); hub.cache_token_ids = [77]
    target.past_key_values = _fake_past(1); target.cache_token_ids = [77]
    source.past_key_values = _fake_past(3); source.cache_token_ids = [1,2,3]
    runner._retain_pending_self_refresh_prefix_tokens[source.node_id] = 2
    runner._retain_hub_sync_tokens_by_agent[source.node_id] = 1

    def fake_retokenize(*, source_model, target_model, source_context_token_ids):
        ids = source_context_token_ids.as_tensor().squeeze(0).detach().cpu().tolist()
        # Only the newly committed suffix is ever retokenized in either direction.
        if source_model.id != target_model.id:
            if source_model.id == source.model.id:
                assert ids == [3]
                return TokenIDs(torch.tensor([[78]], dtype=torch.long), model_id=target_model.id)
            assert ids == [78]
            return TokenIDs(torch.tensor([[3]], dtype=torch.long), model_id=target_model.id)
        return TokenIDs(torch.tensor([ids], dtype=torch.long), model_id=target_model.id)

    monkeypatch.setattr("alg.mot.train.retokenize_agent_runner_context", fake_retokenize)
    monkeypatch.setattr(
        runner.cache_translator,
        "_build_algorithm_translated_past",
        lambda **kwargs: _fake_past(int(kwargs["target_context_token_ids"].shape[1])),
    )

    runner._prepare_outgoing_route_translation(source_agent=source, logical_target_agent=target)
    _, first_meta, _, _, second_meta = runner._star_offload_to_agent(
        source_agent=source, target_agent=target
    )

    assert first_meta["append_only_cross_tokenizer"] is True
    assert first_meta["delta_prefix_matched"] is True
    assert first_meta["target_tokens_before_replay"] == 1
    assert hub.cache_token_ids == [77, 78]
    # The old same-model C prefix remains byte/token identical and receives only
    # the new Hub suffix; there is no stale-prefix replacement path.
    assert second_meta["delta_prefix_matched"] is True
    assert target.cache_token_ids == [77, 78]
    assert target.cache_seq_len == 2
    # Cross self-refresh is deferred until the next target has generated. It also
    # replaces only B's native Memory suffix, never B's historical prefix.
    assert runner._retain_hub_sync_tokens_by_agent[source.node_id] == 1
    runner._flush_retain_nonhub_source_cache_refreshes()
    assert source.cache_token_ids == [1, 2, 3]
    assert runner._retain_hub_sync_tokens_by_agent[source.node_id] == 2
    assert runner._retain_hub_sync_tokens_by_agent[target.node_id] == 2



def test_initial_verification_retry_exhaustion_marks_agent_failure_without_fallback(monkeypatch) -> None:
    import core.agent_runner as agent_runner_module

    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        log_agents=False,
    )
    agent = runner.hub_agent
    runner._update_peak_memory_breakdown = lambda: None
    runner._verify_response_semantics_with_model = lambda **kwargs: ("always reject", False)
    monkeypatch.setattr(agent_runner_module, "VERIFICATION_MAX_RETRIES", 2)

    calls = []
    responses = [
        "The final answer is Yes.",
        "Retry one still says Yes.",
        "Retry two still says Yes.",
    ]

    def fake_generate(prompt_text: str) -> AgentGeneration:
        text = responses[len(calls)]
        calls.append(text)
        return AgentGeneration(
            agent_id=agent.node_id,
            prompt_text=prompt_text,
            text=text,
            raw_text=text,
            generated_token_ids=[len(calls)],
            tokens_before=0,
            tokens_after=1,
            tokens_prompt=1,
            tokens_completion=1,
        )

    agent.generate_response = fake_generate
    generation, verification, attempts, counts, _ = runner._generate_verified_initial(
        agent=agent,
        initial_prompt="initial",
        context="",
        question="Question?",
    )

    assert len(calls) == 3  # original + two retries
    assert attempts == 2
    assert generation.text == responses[-1]
    assert verification.passed is False
    assert verification.final_answer is None
    assert "agent excluded from future discussion and assigned permanent Failed vote" in verification.reason
    assert counts.syntax_failures == 3


def test_followup_verification_retry_exhaustion_marks_agent_failure_without_fallback(monkeypatch) -> None:
    import core.agent_runner as agent_runner_module

    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        log_agents=False,
    )
    agent = runner.agent_sequence[1]
    runner._update_peak_memory_breakdown = lambda: None
    runner._verify_response_semantics_with_model = lambda **kwargs: ("stance=No; final=Yes", False)
    monkeypatch.setattr(agent_runner_module, "VERIFICATION_MAX_RETRIES", 2)

    calls = []
    responses = [
        "[AGREE] Original response.\nFinal Solution: Yes",
        "[AGREE] Retry one.\nFinal Solution: Yes",
        "[AGREE] Retry two.\nFinal Solution: Yes",
    ]

    def fake_generate(prompt_text: str) -> AgentGeneration:
        text = responses[len(calls)]
        calls.append(text)
        return AgentGeneration(
            agent_id=agent.node_id,
            prompt_text=prompt_text,
            text=text,
            raw_text=text,
            generated_token_ids=[len(calls)],
            tokens_before=0,
            tokens_after=1,
            tokens_prompt=1,
            tokens_completion=1,
        )

    agent.generate_response = fake_generate
    generation, verification, attempts, counts, _ = runner._generate_verified_followup(
        agent=agent,
        initial_prompt="initial",
        context="",
        question="Question?",
        current_solution="Yes",
        current_answer="Yes",
        current_response="Final Solution: Yes",
        turn_index=2,
        agent_index=1,
    )

    assert len(calls) == 3  # original + two retries
    assert attempts == 2
    assert generation.text == responses[-1]
    assert verification.passed is False
    assert verification.marker == "agree"
    assert verification.final_answer is None
    assert "agent excluded from future discussion and assigned permanent Failed vote" in verification.reason
    assert counts.semantic_failures == 3



def test_verification_retry_default_is_ten() -> None:
    import core.agent_runner as agent_runner_module

    assert agent_runner_module.VERIFICATION_MAX_RETRIES == 10


def test_two_agent_followup_failure_revokes_vote_excludes_agent_and_forces_failed(monkeypatch) -> None:
    import core.agent_runner as agent_runner_module

    monkeypatch.setattr(agent_runner_module, "VERIFICATION_MAX_RETRIES", 2)
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        max_turns=6,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    calls = {"A": 0, "B": 0}

    def fake_semantic(**kwargs):
        return ("ok" if kwargs["agent"].node_id == "A" else "reject", kwargs["agent"].node_id == "A")

    runner._verify_response_semantics_with_model = fake_semantic

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls[agent.node_id] += 1
            text = (
                "A verified history.\nFinal Solution: Yes"
                if agent.node_id == "A"
                else "[DISAGREE] B failed attempt.\nFinal Solution: No"
            )
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
            "edge_id": "edge",
            "offload_kind": "delta",
            "tokens_sent": 1,
            "tokens_received": 1,
            "expected_delta_tokens": 1,
        }
        return target_agent, meta, False, source_agent, meta

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    assert calls == {"A": 1, "B": 3}  # original + two retries for B
    assert result.prediction == "Failed"
    assert result.accuracy == 0.0
    assert result.profile["final_decision_method"] == "failed_consensus"
    assert result.profile["failed_agent_ids"] == ["B"]
    assert result.profile["active_agent_ids"] == ["A"]
    assert result.profile["final_agent_votes"] == {"A": "Yes", "B": "Failed"}
    assert result.profile["final_vote_counts"] == {"Yes": 1, "No": 0, "Failed": 1}
    assert len(result.agent_messages) == 2
    failed = result.agent_messages[1]
    assert failed.agent_id == "B"
    assert failed.agent_failed is True
    assert failed.response_state == "failed"
    assert failed.verification_passed is False
    assert failed.final_answer is None
    assert "A verified history." in result.transcript
    assert "B failed attempt." not in result.transcript


def test_failed_agent_keeps_old_history_but_is_skipped_on_future_cycles(monkeypatch) -> None:
    import core.agent_runner as agent_runner_module

    monkeypatch.setattr(agent_runner_module, "VERIFICATION_MAX_RETRIES", 1)
    ctx = _ctx("tiny-a,tiny-b")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=5,
        max_turns=9,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    calls = {agent.node_id: 0 for agent in runner.agent_sequence}
    answers = {"A": "Yes", "B": "No", "C": "Yes", "D": "No", "E": "No"}

    def fake_semantic(**kwargs):
        agent_id = kwargs["agent"].node_id
        # B succeeds on its first discussion turn, then every attempt on its
        # second turn is rejected until the retry budget is exhausted.
        if agent_id == "B" and calls["B"] >= 2:
            return "reject", False
        return "ok", True

    runner._verify_response_semantics_with_model = fake_semantic

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls[agent.node_id] += 1
            if agent.node_id == "A" and calls["A"] == 1:
                text = "A initial.\nFinal Solution: Yes"
            else:
                text = f"[DISAGREE] {agent.node_id} response {calls[agent.node_id]}.\nFinal Solution: {answers[agent.node_id]}"
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
    prepared_routes = []

    def fake_prepare(*, source_agent, logical_target_agent):
        prepared_routes.append((source_agent.node_id, logical_target_agent.node_id))

    runner._prepare_outgoing_route_translation = fake_prepare
    runner._update_peak_memory_breakdown = lambda: None
    offload_routes = []

    def fake_star_offload(*, source_agent, target_agent):
        offload_routes.append((source_agent.node_id, target_agent.node_id))
        meta = {
            "edge_id": "edge",
            "offload_kind": "delta",
            "tokens_sent": 1,
            "tokens_received": 1,
            "expected_delta_tokens": 1,
        }
        return target_agent, meta, False, source_agent, meta

    runner._star_offload_to_agent = fake_star_offload
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    assert [message.agent_id for message in result.agent_messages] == [
        "A", "B", "C", "D", "E", "A", "B", "C", "D"
    ]
    first_b = result.agent_messages[1]
    failed_b = result.agent_messages[6]
    assert first_b.agent_failed is False
    assert failed_b.agent_failed is True
    assert failed_b.final_answer is None
    assert result.profile["failed_agent_ids"] == ["B"]
    assert "B response 1." in result.transcript
    assert "B response 2." not in result.transcript
    assert result.profile["final_agent_votes"]["B"] == "Failed"
    assert result.profile["active_agent_ids"] == ["A", "C", "D", "E"]
    # B's second-cycle verification failure commits no Memory. The next canonical
    # route must therefore stay hub-originated; routing B's translated copy back
    # through the hub would re-tokenize unchanged history and can corrupt the
    # retained prefix on heterogeneous model grids.
    assert prepared_routes.count(("B", "C")) == 1  # first-cycle B success only
    assert offload_routes.count(("B", "C")) == 1   # no post-failure B handoff
    assert ("A", "C") in prepared_routes
    assert ("A", "C") in offload_routes
    assert prepared_routes[-2:] == [("A", "C"), ("C", "D")]


def test_initial_agent_failure_in_two_agent_run_forces_failed_without_trying_lone_agent(monkeypatch) -> None:
    import core.agent_runner as agent_runner_module

    monkeypatch.setattr(agent_runner_module, "VERIFICATION_MAX_RETRIES", 1)
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        max_turns=4,
        log_agents=False,
    )
    runner._generate_expert_personas = lambda context, question: {
        agent.node_id: (f"Expert {agent.node_id}", "Useful expert.") for agent in runner.agent_sequence
    }
    runner._verify_response_semantics_with_model = lambda **kwargs: ("reject", False)
    calls = {"A": 0, "B": 0}

    def fake_generate(agent):
        def generate(prompt_text: str) -> AgentGeneration:
            calls[agent.node_id] += 1
            text = "Unverified.\nFinal Solution: Yes"
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
    runner._update_peak_memory_breakdown = lambda: None
    result = runner.run(context="", question="Question?", gold_answers=["Yes"])

    assert calls == {"A": 2, "B": 0}
    assert result.prediction == "Failed"
    assert result.accuracy == 0.0
    assert result.profile["failed_agent_ids"] == ["A"]
    assert result.profile["active_agent_ids"] == ["B"]
    assert result.agent_messages[0].agent_failed is True
    assert result.transcript == ""



def test_failed_vote_replaces_previous_vote_and_stays_in_original_denominator() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=8,
        log_agents=False,
    )
    votes = {
        "A": "Yes", "B": "Yes", "C": "Yes", "D": "Yes",
        "E": "Yes", "F": "Yes", "G": "No", "H": "No",
    }
    assert runner._supermajority_consensus(votes) == "Yes"

    runner._mark_agent_failed("A", votes)
    assert votes["A"] == "Failed"
    assert runner._supermajority_consensus(votes) is None
    assert runner._active_discussion_agent_count() == 7
    assert runner._should_force_failed_result() is False


def test_failed_vote_early_termination_matches_strict_supermajority_boundary() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner8 = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=8,
        log_agents=False,
    )
    votes8 = {}
    runner8._mark_agent_failed("A", votes8)
    runner8._mark_agent_failed("B", votes8)
    # 6/8 = 75% remains available, so Yes/No >66% is still possible.
    assert runner8._should_force_failed_result() is False
    runner8._mark_agent_failed("C", votes8)
    # Only 5/8 = 62.5% remain: no future Yes/No vote can exceed 66%.
    assert runner8._should_force_failed_result() is True

    turns = []
    answer = runner8._evaluate_turn_consensus(
        turn_index=3,
        agent_id="C",
        agent_votes=votes8,
        turns=turns,
    )
    assert answer == "Failed"
    assert turns[-1].consensus_reached is True
    assert turns[-1].consensus_answer == "Failed"
    assert turns[-1].vote_counts == {"Yes": 0, "No": 0, "Failed": 3}

    runner3 = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=3,
        log_agents=False,
    )
    votes3 = {}
    runner3._mark_agent_failed("A", votes3)
    # Strict >0.66 still allows 2/3 ~= 0.6667, so one failure must not stop 3 agents.
    assert runner3._should_force_failed_result() is False
    runner3._mark_agent_failed("B", votes3)
    assert runner3._should_force_failed_result() is True


def test_final_majority_vote_counts_failed_as_a_real_vote() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=5,
        log_agents=False,
    )
    answer, method = runner._majority_vote_with_random_tie(
        {"A": "Failed", "B": "Failed", "C": "Yes", "D": "No"}
    )
    assert answer == "Failed"
    assert method == "majority_vote"


def test_retain_mot_self_refresh_replaces_only_native_memory_suffix(monkeypatch) -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        cache_mode="retain",
        log_agents=False,
    )
    hub = runner.hub_agent
    source = runner.agents["B"]
    prefix_tokens = 3
    full_ids = [10, 11, 12, 20, 21]

    def valued_past(values):
        values = torch.tensor(values, dtype=torch.float32).view(1, 1, -1, 1)
        return tuple((values.clone(), values.clone()) for _ in range(2))

    # Hub already received B's new Memory and therefore owns the canonical source
    # representation for the complete five-token history.
    hub.past_key_values = valued_past([1, 2, 3, 4, 5, 6, 7])
    hub.cache_token_ids = list(full_ids) + [30, 31]
    # B's old prefix is canonical, but its newest Memory suffix was appended by a
    # normal native forward and deliberately differs from the MoT replay suffix.
    source.past_key_values = valued_past([31, 32, 33, 70, 71])
    source.cache_token_ids = list(full_ids)
    runner._retain_pending_self_refresh_prefix_tokens[source.node_id] = prefix_tokens

    monkeypatch.setattr(
        runner.cache_translator,
        "target_token_ids_for_source",
        lambda **kwargs: list(full_ids),
    )
    seen = {}

    def fake_build(**kwargs):
        seen["source_tokens"] = list(kwargs["source_token_ids"])
        seen["full_tokens"] = list(kwargs["retain_source_full_token_ids"])
        seen["prefix_len"] = get_past_seq_len(kwargs["retain_target_prefix_past_key_values"])
        return "A_to_B", valued_past([90, 91]), full_ids[prefix_tokens:]

    monkeypatch.setattr(runner.cache_translator, "build_pretranslated_past_for_edge", fake_build)

    runner._refresh_retain_nonhub_source_cache_from_hub(source, hub_context_tokens=len(full_ids))

    assert seen == {
        "source_tokens": full_ids[prefix_tokens:],
        "full_tokens": full_ids,
        "prefix_len": prefix_tokens,
    }
    assert source.cache_token_ids == full_ids
    key = source.past_key_values[0][0].reshape(-1).tolist()
    assert key == [31.0, 32.0, 33.0, 90.0, 91.0]
    assert source.node_id not in runner._retain_pending_self_refresh_prefix_tokens


def test_retain_mot_self_refresh_is_deferred_past_current_handoff(monkeypatch) -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        cache_mode="retain",
        log_agents=False,
    )
    source = runner.agents["B"]
    target = runner.hub_agent
    order = []
    _install_fake_cache(target, [1, 2, 3])
    runner._retain_pending_self_refresh_prefix_tokens[source.node_id] = 1

    def fake_offload(*, source_agent, target_agent, **kwargs):
        del kwargs
        order.append(("offload", source_agent.node_id, target_agent.node_id))
        return {"edge_id": "B_to_A", "tokens_sent": 1, "expected_delta_tokens": 1}, False

    def fake_refresh(agent, *, hub_context_tokens=None):
        order.append(("refresh", agent.node_id, hub_context_tokens))
        runner._retain_deferred_self_refresh_hub_tokens.pop(agent.node_id, None)
        runner._retain_pending_self_refresh_prefix_tokens.pop(agent.node_id, None)

    monkeypatch.setattr(runner, "_offload_delta_hop", fake_offload)
    monkeypatch.setattr(runner, "_refresh_retain_nonhub_source_cache_from_hub", fake_refresh)

    runner._star_offload_to_agent(source_agent=source, target_agent=target)

    # The physical handoff only schedules refresh.  No retained source mutation is
    # allowed until the next target's generation/verification has finished.
    assert order == [("offload", "B", "A")]
    assert runner._retain_deferred_self_refresh_hub_tokens == {"B": 3}

    runner._flush_retain_nonhub_source_cache_refreshes()
    assert order == [("offload", "B", "A"), ("refresh", "B", 3)]


def test_free_mot_revisit_rebuilds_historical_retain_chunks(monkeypatch) -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        cache_mode="free",
        log_agents=False,
    )
    hub = runner.hub_agent
    target = runner.agents["B"]
    source_ids = list(range(10))
    _install_fake_cache(hub, source_ids)
    target.clear_kv_cache()
    # B's retained counterpart was historically built as:
    #   full [0:3] -> incremental [3:5] -> current incremental [5:10].
    runner._mot_replay_boundaries_by_agent[target.node_id] = [3, 5]

    seen = []

    def fake_build(*, source_agent, target_agent, source_past_key_values, source_token_ids, **kwargs):
        seen.append({
            "ids": list(source_token_ids),
            "past_tokens": get_past_seq_len(source_past_key_values),
            "full_ids": list(kwargs.get("retain_source_full_token_ids", [])),
            "full_past_tokens": (
                get_past_seq_len(kwargs["retain_source_full_past_key_values"])
                if kwargs.get("retain_source_full_past_key_values") is not None else 0
            ),
            "target_prefix_tokens": (
                get_past_seq_len(kwargs["retain_target_prefix_past_key_values"])
                if kwargs.get("retain_target_prefix_past_key_values") is not None else 0
            ),
        })
        edge = runner.cache_translator._get_edge(source_agent.node_id, target_agent.node_id)
        return edge.id, _fake_past(len(source_token_ids)), list(source_token_ids)

    monkeypatch.setattr(runner.cache_translator, "build_pretranslated_past_for_edge", fake_build)

    runner._prepare_outgoing_route_translation(source_agent=hub, logical_target_agent=target)

    assert seen == [
        {
            "ids": [0, 1, 2],
            "past_tokens": 3,
            "full_ids": [],
            "full_past_tokens": 0,
            "target_prefix_tokens": 0,
        },
        {
            "ids": [3, 4],
            "past_tokens": 2,
            "full_ids": [0, 1, 2, 3, 4],
            "full_past_tokens": 5,
            "target_prefix_tokens": 3,
        },
        {
            "ids": [5, 6, 7, 8, 9],
            "past_tokens": 5,
            "full_ids": source_ids,
            "full_past_tokens": 10,
            "target_prefix_tokens": 5,
        },
    ]

    _, incoming_meta, _, _, _ = runner._star_offload_to_agent(source_agent=hub, target_agent=target)
    assert incoming_meta["tokens_sent"] == 10
    assert target.cache_token_ids == source_ids
    assert target.cache_seq_len == 10


def test_mot_replay_boundaries_survive_free_cache_clear() -> None:
    ctx = _ctx("tiny-a,tiny-a")
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=2,
        cache_mode="free",
        log_agents=False,
    )
    target = runner.agents["B"]
    _install_fake_cache(target, [1, 2, 3])
    runner._record_mot_replay_boundary(target)
    target.clear_kv_cache()
    assert target.past_key_values is None
    assert runner._mot_replay_boundaries_by_agent[target.node_id] == [3]


def test_free_and_retain_two_agent_retry_exhaustion_force_same_failed_result(monkeypatch) -> None:
    """Retry exhaustion must have identical discussion semantics in Free/Retain.

    Cache transport/accounting may differ between the two modes, but once B exhausts
    verification both controllers must revoke B, keep A's old history, and force the
    same terminal Failed result because only one discussion Agent remains.
    """
    import core.agent_runner as agent_runner_module

    monkeypatch.setattr(agent_runner_module, "VERIFICATION_MAX_RETRIES", 1)

    def run_mode(cache_mode: str):
        ctx = _ctx("tiny-a,tiny-a")
        runner = AgentRunner(
            ctx=ctx,
            translator_pool=ctx.tp,
            alg="mot",
            agent_count=2,
            max_turns=6,
            cache_mode=cache_mode,
            log_agents=False,
        )
        runner._generate_expert_personas = lambda context, question: {
            agent.node_id: (f"Expert {agent.node_id}", "Useful expert.")
            for agent in runner.agent_sequence
        }
        calls = {"A": 0, "B": 0}
        runner._verify_response_semantics_with_model = lambda **kwargs: (
            ("ok", True) if kwargs["agent"].node_id == "A" else ("reject", False)
        )

        def fake_generate(agent):
            def generate(prompt_text: str) -> AgentGeneration:
                calls[agent.node_id] += 1
                text = (
                    "A verified history.\nFinal Solution: Yes"
                    if agent.node_id == "A"
                    else f"[DISAGREE] B failed attempt {calls['B']}.\nFinal Solution: No"
                )
                return AgentGeneration(
                    agent_id=agent.node_id,
                    prompt_text=prompt_text,
                    text=text,
                    raw_text=text,
                    generated_token_ids=[calls[agent.node_id]],
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

        transport_tokens = 9 if cache_mode == "free" else 3

        def fake_star_offload(*, source_agent, target_agent):
            meta = {
                "edge_id": f"{source_agent.node_id}_to_{target_agent.node_id}",
                "offload_kind": "delta",
                "tokens_sent": transport_tokens,
                "tokens_received": transport_tokens,
                "expected_delta_tokens": transport_tokens,
            }
            source_cleared = cache_mode == "free" and source_agent.node_id != runner.hub_agent.node_id
            return target_agent, meta, source_cleared, source_agent, meta

        runner._star_offload_to_agent = fake_star_offload
        return runner.run(context="", question="Question?", gold_answers=["Yes"])

    free = run_mode("free")
    retain = run_mode("retain")

    def semantic_signature(result):
        return {
            "prediction": result.prediction,
            "accuracy": result.accuracy,
            "transcript": result.transcript,
            "messages": [
                (
                    message.agent_id,
                    message.prompt,
                    message.response,
                    message.response_state,
                    message.verification_attempts,
                    message.verification_passed,
                    message.final_answer,
                    message.agent_failed,
                    message.solution,
                )
                for message in result.agent_messages
            ],
            "turns": [
                (
                    turn.turn_index,
                    turn.agent_id,
                    turn.agent_votes,
                    turn.vote_counts,
                    turn.consensus_reached,
                    turn.consensus_answer,
                )
                for turn in result.turns
            ],
            "failed": result.profile["failed_agent_ids"],
            "active": result.profile["active_agent_ids"],
            "final_votes": result.profile["final_agent_votes"],
            "decision": result.profile["final_decision_method"],
        }

    assert semantic_signature(free) == semantic_signature(retain)
    assert free.prediction == retain.prediction == "Failed"
    assert free.accuracy == retain.accuracy == 0.0
    assert free.profile["failed_agent_ids"] == retain.profile["failed_agent_ids"] == ["B"]
    assert free.profile["active_agent_ids"] == retain.profile["active_agent_ids"] == ["A"]
    # Deliberately different transport accounting proves the equality assertion is
    # about discussion semantics, not about making Free/Retain cache mechanics equal.
    assert free.agent_messages[0].tokens_sent != retain.agent_messages[0].tokens_sent


def test_free_and_retain_midcycle_retry_exhaustion_keep_later_discussion_identical(monkeypatch) -> None:
    """A failed revisiting Agent must not make later Free/Retain discussion diverge."""
    import core.agent_runner as agent_runner_module

    monkeypatch.setattr(agent_runner_module, "VERIFICATION_MAX_RETRIES", 1)

    def run_mode(cache_mode: str):
        ctx = _ctx("tiny-a,tiny-a")
        runner = AgentRunner(
            ctx=ctx,
            translator_pool=ctx.tp,
            alg="mot",
            agent_count=5,
            max_turns=9,
            cache_mode=cache_mode,
            log_agents=False,
        )
        runner._generate_expert_personas = lambda context, question: {
            agent.node_id: (f"Expert {agent.node_id}", "Useful expert.")
            for agent in runner.agent_sequence
        }
        calls = {agent.node_id: 0 for agent in runner.agent_sequence}
        answers = {"A": "Yes", "B": "No", "C": "Yes", "D": "No", "E": "No"}

        def fake_semantic(**kwargs):
            agent_id = kwargs["agent"].node_id
            # B succeeds in cycle 1, but on its cycle-2 turn both the original
            # response and one corrective retry are rejected.
            if agent_id == "B" and calls["B"] >= 2:
                return "reject", False
            return "ok", True

        runner._verify_response_semantics_with_model = fake_semantic

        def fake_generate(agent):
            def generate(prompt_text: str) -> AgentGeneration:
                calls[agent.node_id] += 1
                if agent.node_id == "A" and calls["A"] == 1:
                    text = "A initial.\nFinal Solution: Yes"
                else:
                    text = (
                        f"[DISAGREE] {agent.node_id} response {calls[agent.node_id]}.\n"
                        f"Final Solution: {answers[agent.node_id]}"
                    )
                return AgentGeneration(
                    agent_id=agent.node_id,
                    prompt_text=prompt_text,
                    text=text,
                    raw_text=text,
                    generated_token_ids=[calls[agent.node_id]],
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
        transport_tokens = 11 if cache_mode == "free" else 2

        def fake_star_offload(*, source_agent, target_agent):
            meta = {
                "edge_id": f"{source_agent.node_id}_to_{target_agent.node_id}",
                "offload_kind": "delta",
                "tokens_sent": transport_tokens,
                "tokens_received": transport_tokens,
                "expected_delta_tokens": transport_tokens,
            }
            source_cleared = cache_mode == "free" and source_agent.node_id != runner.hub_agent.node_id
            return target_agent, meta, source_cleared, source_agent, meta

        runner._star_offload_to_agent = fake_star_offload
        return runner.run(context="", question="Question?", gold_answers=["Yes"])

    free = run_mode("free")
    retain = run_mode("retain")

    def semantic_signature(result):
        return {
            "prediction": result.prediction,
            "accuracy": result.accuracy,
            "transcript": result.transcript,
            "messages": [
                (
                    message.agent_id,
                    message.prompt,
                    message.response,
                    message.response_state,
                    message.verification_attempts,
                    message.verification_syntax_failures,
                    message.verification_semantic_failures,
                    message.verification_passed,
                    message.final_answer,
                    message.agent_failed,
                    message.solution,
                )
                for message in result.agent_messages
            ],
            "turns": [
                (
                    turn.turn_index,
                    turn.agent_id,
                    turn.agent_votes,
                    turn.vote_counts,
                    turn.consensus_reached,
                    turn.consensus_answer,
                )
                for turn in result.turns
            ],
            "failed": result.profile["failed_agent_ids"],
            "active": result.profile["active_agent_ids"],
            "final_votes": result.profile["final_agent_votes"],
            "decision": result.profile["final_decision_method"],
        }

    assert semantic_signature(free) == semantic_signature(retain)
    assert [message.agent_id for message in free.agent_messages] == [
        "A", "B", "C", "D", "E", "A", "B", "C", "D"
    ]
    failed_b = free.agent_messages[6]
    assert failed_b.agent_failed is True
    assert failed_b.final_answer is None
    assert "B response 1." in free.transcript
    assert "B response 2." not in free.transcript
    assert free.profile["failed_agent_ids"] == retain.profile["failed_agent_ids"] == ["B"]
    assert free.profile["active_agent_ids"] == retain.profile["active_agent_ids"] == ["A", "C", "D", "E"]
    assert free.profile["final_agent_votes"]["B"] == "Failed"
    assert retain.profile["final_agent_votes"]["B"] == "Failed"
    # Continue beyond B's failed turn and prove transport mechanics may differ while
    # the following C/D discussion remains byte-for-byte the same.
    assert free.agent_messages[7].response == retain.agent_messages[7].response
    assert free.agent_messages[8].response == retain.agent_messages[8].response
    assert any(
        free_message.tokens_received != retain_message.tokens_received
        for free_message, retain_message in zip(free.agent_messages, retain.agent_messages)
    )


def test_free_and_retain_three_of_eight_failures_terminate_at_same_turn(monkeypatch) -> None:
    """The 3/8 Failed boundary must terminate Free/Retain identically."""
    import core.agent_runner as agent_runner_module

    monkeypatch.setattr(agent_runner_module, "VERIFICATION_MAX_RETRIES", 1)

    def run_mode(cache_mode: str):
        ctx = _ctx("tiny-a,tiny-a")
        runner = AgentRunner(
            ctx=ctx,
            translator_pool=ctx.tp,
            alg="mot",
            agent_count=8,
            max_turns=16,
            cache_mode=cache_mode,
            log_agents=False,
        )
        runner._generate_expert_personas = lambda context, question: {
            agent.node_id: (f"Expert {agent.node_id}", "Useful expert.")
            for agent in runner.agent_sequence
        }
        calls = {agent.node_id: 0 for agent in runner.agent_sequence}
        failing = {"B", "D", "F"}

        def fake_semantic(**kwargs):
            return ("reject", False) if kwargs["agent"].node_id in failing else ("ok", True)

        runner._verify_response_semantics_with_model = fake_semantic

        def fake_generate(agent):
            def generate(prompt_text: str) -> AgentGeneration:
                calls[agent.node_id] += 1
                if agent.node_id == "A" and calls["A"] == 1:
                    text = "A initial.\nFinal Solution: Yes"
                else:
                    answer = "Yes" if agent.node_id in {"C", "E", "G"} else "No"
                    text = f"[DISAGREE] {agent.node_id} response.\nFinal Solution: {answer}"
                return AgentGeneration(
                    agent_id=agent.node_id,
                    prompt_text=prompt_text,
                    text=text,
                    raw_text=text,
                    generated_token_ids=[calls[agent.node_id]],
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
        transport_tokens = 13 if cache_mode == "free" else 4

        def fake_star_offload(*, source_agent, target_agent):
            meta = {
                "edge_id": f"{source_agent.node_id}_to_{target_agent.node_id}",
                "offload_kind": "delta",
                "tokens_sent": transport_tokens,
                "tokens_received": transport_tokens,
                "expected_delta_tokens": transport_tokens,
            }
            source_cleared = cache_mode == "free" and source_agent.node_id != runner.hub_agent.node_id
            return target_agent, meta, source_cleared, source_agent, meta

        runner._star_offload_to_agent = fake_star_offload
        result = runner.run(context="", question="Question?", gold_answers=["Yes"])
        return result, calls

    free, free_calls = run_mode("free")
    retain, retain_calls = run_mode("retain")

    assert free_calls == retain_calls
    assert [m.agent_id for m in free.agent_messages] == ["A", "B", "C", "D", "E", "F"]
    assert [m.agent_id for m in retain.agent_messages] == ["A", "B", "C", "D", "E", "F"]
    assert free.prediction == retain.prediction == "Failed"
    assert free.accuracy == retain.accuracy == 0.0
    assert free.profile["consensus_turn"] == retain.profile["consensus_turn"] == 6
    assert free.profile["consensus_answer"] == retain.profile["consensus_answer"] == "Failed"
    assert free.profile["final_decision_method"] == retain.profile["final_decision_method"] == "failed_consensus"
    assert free.profile["failed_agent_ids"] == retain.profile["failed_agent_ids"] == ["B", "D", "F"]
    assert free.profile["final_agent_votes"] == retain.profile["final_agent_votes"]
    assert free.profile["final_vote_counts"] == retain.profile["final_vote_counts"] == {
        "Yes": 3,
        "No": 0,
        "Failed": 3,
    }
    assert free.turns[-1].vote_counts == retain.turns[-1].vote_counts == {
        "Yes": 3,
        "No": 0,
        "Failed": 3,
    }
    assert free.turns[-1].consensus_reached is True
    assert retain.turns[-1].consensus_reached is True
    # G/H never speak because after F fails only 5/8 vote slots remain available,
    # so strict >0.66 Yes/No consensus is impossible.
    assert free_calls["G"] == free_calls["H"] == 0
    assert retain_calls["G"] == retain_calls["H"] == 0


def test_heterogeneous_retain_hub_revisit_appends_only_unsynced_hub_suffix(monkeypatch) -> None:
    from core.common import TokenIDs

    ctx = Context(SimpleNamespace(model_ids="tiny-a,tiny-b", model_directions="all", device="cpu", dtype="float32"))
    runner = AgentRunner(ctx=ctx, translator_pool=ctx.tp, alg="mot", agent_count=2, cache_mode="retain", log_agents=False)
    hub, target = runner.hub_agent, runner.agents["B"]
    hub.past_key_values = _fake_past(3); hub.cache_token_ids = [70, 71, 72]
    target.past_key_values = _fake_past(2); target.cache_token_ids = [10, 11]
    runner._retain_hub_sync_tokens_by_agent[target.node_id] = 2
    seen_source_rows = []

    def fake_retokenize(*, source_model, target_model, source_context_token_ids):
        ids = source_context_token_ids.as_tensor().squeeze(0).detach().cpu().tolist()
        seen_source_rows.append(ids)
        assert ids == [72]  # never [70,71,72]
        return TokenIDs(torch.tensor([[12, 13]], dtype=torch.long), model_id=target_model.id)

    monkeypatch.setattr("alg.mot.train.retokenize_agent_runner_context", fake_retokenize)
    monkeypatch.setattr(
        runner.cache_translator,
        "_build_algorithm_translated_past",
        lambda **kwargs: _fake_past(int(kwargs["target_context_token_ids"].shape[1])),
    )

    runner._prepare_outgoing_route_translation(source_agent=hub, logical_target_agent=target)
    _, meta, _, _, _ = runner._star_offload_to_agent(source_agent=hub, target_agent=target)

    assert seen_source_rows == [[72]]
    assert meta["append_only_cross_tokenizer"] is True
    assert meta["target_tokens_before_replay"] == 2
    assert target.cache_token_ids == [10, 11, 12, 13]
    assert hub.cache_token_ids == [70, 71, 72]
    assert runner._retain_hub_sync_tokens_by_agent[target.node_id] == 3


def _past_scalar_values(past_key_values):
    return past_key_values[0][0][0, 0, :, 0].detach().cpu().tolist()


def test_free_and_retain_heterogeneous_b_to_hub_to_c_keep_identical_canonical_kv(monkeypatch) -> None:
    """Regression for the real Turn-3 divergence: A(Llama)->B(Qwen)->A->C(Llama).

    B contributes only one new Memory suffix. Free and Retain must append exactly
    that translated suffix to the resident canonical Hub prefix. A full B-history
    decode/re-encode in Free would rewrite the Hub prefix and make C diverge.
    """
    from core.common import TokenIDs

    snapshots = {}

    for mode in ("free", "retain"):
        ctx = Context(SimpleNamespace(model_ids="tiny-a,tiny-b", model_directions="all", device="cpu", dtype="float32"))
        runner = AgentRunner(
            ctx=ctx,
            translator_pool=ctx.tp,
            alg="mot",
            agent_count=3,
            cache_mode=mode,
            log_agents=False,
        )
        hub, source, target = runner.hub_agent, runner.agents["B"], runner.agents["C"]

        def valued(values):
            tensor = torch.tensor(values, dtype=torch.float32).view(1, 1, -1, 1)
            return ((tensor.clone(), tensor.clone()),)

        # Before B speaks, A's canonical history is one Llama token [77]. B owns
        # the equivalent Qwen-grid prefix [1,2], then natively appends Memory [3].
        hub.past_key_values = valued([77]); hub.cache_token_ids = [77]
        source.past_key_values = valued([1, 2, 3]); source.cache_token_ids = [1, 2, 3]
        target.clear_kv_cache()
        runner._retain_pending_self_refresh_prefix_tokens[source.node_id] = 2
        if mode == "retain":
            runner._retain_hub_sync_tokens_by_agent[source.node_id] = 1

        def fake_retokenize(*, source_model, target_model, source_context_token_ids):
            ids = source_context_token_ids.as_tensor().squeeze(0).detach().cpu().tolist()
            if source_model.id == target_model.id:
                out = ids
            elif source_model.id == source.model.id and target_model.id == hub.model.id:
                # Correct append-only route sees only B's new Memory [3]. The old
                # Free bug retokenized [1,2,3] and would replace Hub [77].
                out = [78] if ids == [3] else [700, 701, 78]
            elif source_model.id == hub.model.id and target_model.id == source.model.id:
                out = [3] if ids == [78] else [1, 2, 3]
            else:
                out = ids
            return TokenIDs(torch.tensor([out], dtype=torch.long), model_id=target_model.id)

        monkeypatch.setattr("alg.mot.train.retokenize_agent_runner_context", fake_retokenize)

        def fake_algorithm(**kwargs):
            target_ids = kwargs["target_context_token_ids"].as_tensor().squeeze(0).detach().cpu().tolist()
            return valued(target_ids)

        monkeypatch.setattr(runner.cache_translator, "_build_algorithm_translated_past", fake_algorithm)

        runner._prepare_outgoing_route_translation(source_agent=source, logical_target_agent=target)
        _, first_meta, _, _, second_meta = runner._star_offload_to_agent(
            source_agent=source, target_agent=target
        )

        snapshots[mode] = {
            "hub_ids": list(hub.cache_token_ids),
            "hub_values": _past_scalar_values(hub.past_key_values),
            "target_ids": list(target.cache_token_ids),
            "target_values": _past_scalar_values(target.past_key_values),
            "first_tokens": first_meta["tokens_sent"],
            "second_tokens": second_meta["tokens_received"],
        }

    assert snapshots["free"] == snapshots["retain"]
    assert snapshots["free"]["hub_ids"] == [77, 78]
    assert snapshots["free"]["hub_values"] == [77.0, 78.0]
    assert snapshots["free"]["target_ids"] == [77, 78]
    assert snapshots["free"]["target_values"] == [77.0, 78.0]


def test_free_cross_revisit_replays_same_hub_chunks_as_retain(monkeypatch) -> None:
    """Freed Qwen cache must be rebuilt using Retain's historical Hub chunking."""
    from core.common import TokenIDs

    results = {}
    for mode in ("free", "retain"):
        ctx = Context(SimpleNamespace(model_ids="tiny-a,tiny-b", model_directions="all", device="cpu", dtype="float32"))
        runner = AgentRunner(
            ctx=ctx,
            translator_pool=ctx.tp,
            alg="mot",
            agent_count=2,
            cache_mode=mode,
            log_agents=False,
        )
        hub, target = runner.hub_agent, runner.agents["B"]
        hub.past_key_values = _fake_past(3); hub.cache_token_ids = [70, 71, 72]

        # Retain historically built B as Hub[0:1] -> [10,11], then appended
        # Hub[1:2] -> [12]. Free has no resident B KV but remembers the Hub cuts.
        if mode == "retain":
            target.past_key_values = _fake_past(3); target.cache_token_ids = [10, 11, 12]
            runner._retain_hub_sync_tokens_by_agent[target.node_id] = 2
        else:
            target.clear_kv_cache()
        runner._mot_cross_hub_replay_boundaries_by_agent[target.node_id] = [1, 2]

        seen = []

        def fake_retokenize(*, source_model, target_model, source_context_token_ids):
            ids = source_context_token_ids.as_tensor().squeeze(0).detach().cpu().tolist()
            seen.append(ids)
            mapping = {
                (70,): [10, 11],
                (71,): [12],
                (72,): [13, 14],
                # A one-shot full retokenization is deliberately non-canonical.
                (70, 71, 72): [90, 91],
            }
            out = mapping.get(tuple(ids), ids)
            return TokenIDs(torch.tensor([out], dtype=torch.long), model_id=target_model.id)

        monkeypatch.setattr("alg.mot.train.retokenize_agent_runner_context", fake_retokenize)
        monkeypatch.setattr(
            runner.cache_translator,
            "_build_algorithm_translated_past",
            lambda **kwargs: _fake_past(int(kwargs["target_context_token_ids"].shape[1])),
        )

        runner._prepare_outgoing_route_translation(source_agent=hub, logical_target_agent=target)
        _, meta, _, _, _ = runner._star_offload_to_agent(source_agent=hub, target_agent=target)
        results[mode] = {
            "ids": list(target.cache_token_ids),
            "values": _past_scalar_values(target.past_key_values),
            "seen": seen,
            "tokens_received": meta["tokens_received"],
        }

    assert results["free"]["ids"] == results["retain"]["ids"] == [10, 11, 12, 13, 14]
    assert results["free"]["values"] == results["retain"]["values"]
    # Retain needs only the new suffix; Free reconstructs old chunks in the same
    # order and then the same new suffix. Neither mode may canonicalize from the
    # non-canonical one-shot full-history tokenization [90,91].
    assert (70, 71, 72) not in [tuple(x) for x in results["free"]["seen"]]
    assert (70, 71, 72) not in [tuple(x) for x in results["retain"]["seen"]]
    assert results["retain"]["tokens_received"] == 2
    assert results["free"]["tokens_received"] == 5


def test_free_same_grid_revisit_uses_historical_chunks_inside_heterogeneous_pool(monkeypatch) -> None:
    """C/E share Hub's grid even when B/D use another model; replay is per-agent."""
    ctx = Context(SimpleNamespace(model_ids="tiny-a,tiny-b", model_directions="all", device="cpu", dtype="float32"))
    runner = AgentRunner(
        ctx=ctx,
        translator_pool=ctx.tp,
        alg="mot",
        agent_count=3,
        cache_mode="free",
        log_agents=False,
    )
    hub, target = runner.hub_agent, runner.agents["C"]
    assert hub.model.id == target.model.id
    assert runner._homogeneous_model_pool is False
    source_ids = list(range(10))
    _install_fake_cache(hub, source_ids)
    target.clear_kv_cache()
    runner._mot_replay_boundaries_by_agent[target.node_id] = [3, 5]

    seen = []

    def fake_build(*, source_agent, target_agent, source_past_key_values, source_token_ids, **kwargs):
        seen.append((
            list(source_token_ids),
            list(kwargs.get("retain_source_full_token_ids", [])),
            get_past_seq_len(kwargs["retain_target_prefix_past_key_values"])
            if kwargs.get("retain_target_prefix_past_key_values") is not None else 0,
        ))
        edge = runner.cache_translator._get_edge(source_agent.node_id, target_agent.node_id)
        return edge.id, _fake_past(len(source_token_ids)), list(source_token_ids)

    monkeypatch.setattr(runner.cache_translator, "build_pretranslated_past_for_edge", fake_build)

    runner._prepare_outgoing_route_translation(source_agent=hub, logical_target_agent=target)

    assert seen == [
        ([0, 1, 2], [], 0),
        ([3, 4], [0, 1, 2, 3, 4], 3),
        ([5, 6, 7, 8, 9], source_ids, 5),
    ]
