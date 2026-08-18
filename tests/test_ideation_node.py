from langchain_core.messages import AIMessage

from content_agent.nodes.ideation import _AngleSelection, ideation_agent
from content_agent.state import AgentState


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "request": {"client_id": "acme", "channel": "whatsapp", "campaign_topic": "end of season sale"},
        "brand_guidelines": None,
        "brief": None,
        "angle": None,
        "angle_options": None,
        "draft_content": None,
        "compliance_result": None,
        "stage": None,
        "human_decision": None,
        "human_edit_notes": None,
        "final_content": None,
    }
    state.update(overrides)
    return state


def test_ideation_agent_merges_tool_outputs_and_picks_angle():
    result = ideation_agent(_base_state())

    assert result["stage"] == "ideation"
    assert result["angle_options"] == ["stub angle", "stub angle B", "stub angle C"]
    assert result["angle"] == result["angle_options"][0]  # recommendation, first in the list
    # brief passed straight through from brief_creation_tool -- DB fields present
    assert result["brief"]["tone"] == "warm and conversational"
    assert result["brief"]["word_length"] == 60
    assert result["brief"]["key_message"] == "stub key message"
    # guidelines passed straight through from brand_guidelines_tool
    assert result["brand_guidelines"]["prohibited_words"] == ["guarantee", "free money", "act now"]


def test_ideation_agent_includes_edit_notes_in_prompt(monkeypatch):
    captured = {}

    class _RecordingAngleLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return _AngleSelection(angles=["a", "b", "c"])

    monkeypatch.setattr("content_agent.nodes.ideation._structured_llm", _RecordingAngleLLM())

    ideation_agent(_base_state(human_edit_notes="make it more urgent"))

    assert "make it more urgent" in captured["prompt"]
    assert "needed rework" in captured["prompt"]


def test_ideation_agent_no_edit_notes_omits_rejection_language(monkeypatch):
    captured = {}

    class _RecordingAngleLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return _AngleSelection(angles=["a", "b", "c"])

    monkeypatch.setattr("content_agent.nodes.ideation._structured_llm", _RecordingAngleLLM())

    ideation_agent(_base_state())

    assert "needed rework" not in captured["prompt"]


# --- optional tool calling (PRD §11.6) --------------------------------------


class _StubToolLLM:
    """Emits one canned round of tool_calls, then stops."""

    def __init__(self, tool_calls):
        self._tool_calls = tool_calls
        self._invoked = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self._invoked += 1
        if self._invoked == 1 and self._tool_calls:
            return AIMessage(content="", tool_calls=self._tool_calls)
        return AIMessage(content="done")


def test_ideation_agent_reuses_existing_guidelines_and_brief_without_refetching(monkeypatch):
    """On a re-run after an edit (e.g. angle rework), the model can decide
    state's existing brief/guidelines are still current and skip both
    tools -- the node must not fetch fresh copies in that case."""

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not re-fetch -- already present in state")

    monkeypatch.setattr("content_agent.nodes.ideation.brand_guidelines_tool", _fail_if_called)
    monkeypatch.setattr("content_agent.nodes.ideation.brief_creation_tool", _fail_if_called)

    existing_guidelines = {"tone_rules": ["warm"], "prohibited_words": [], "style_guide": "keep it short"}
    existing_brief = {
        "tone": "warm",
        "word_length": 60,
        "key_message": "m",
        "target_audience": "a",
        "constraints": [],
        "cta": "c",
    }

    result = ideation_agent(_base_state(brand_guidelines=existing_guidelines, brief=existing_brief))

    assert result["brand_guidelines"] == existing_guidelines
    assert result["brief"] == existing_brief


def test_ideation_agent_calls_tool_when_model_decides_it_needs_fresh_guidelines(monkeypatch):
    """The model can also call just one of the two tools -- here it fetches
    fresh guidelines but leaves the already-current brief untouched."""
    monkeypatch.setattr(
        "content_agent.nodes.ideation._tool_llm",
        _StubToolLLM([{"name": "fetch_brand_guidelines", "args": {}, "id": "call_1"}]),
    )

    existing_brief = {
        "tone": "warm",
        "word_length": 60,
        "key_message": "m",
        "target_audience": "a",
        "constraints": [],
        "cta": "c",
    }

    result = ideation_agent(_base_state(brand_guidelines=None, brief=existing_brief))

    # real brand_guidelines_tool ran against the seeded DB, replacing the missing state value
    assert result["brand_guidelines"]["prohibited_words"] == ["guarantee", "free money", "act now"]
    assert result["brief"] == existing_brief


def test_ideation_agent_falls_back_to_fetching_when_nothing_exists_and_model_calls_no_tools():
    """Safety net: even if the model makes zero tool calls, the node must
    still end up with real data when state had none to begin with --
    output shouldn't depend on the model reliably remembering to call a
    tool for data it has no substitute for. Uses conftest's default
    _FakeToolLLM, which always makes zero tool calls."""
    result = ideation_agent(_base_state())

    assert result["brand_guidelines"] is not None
    assert result["brief"] is not None
