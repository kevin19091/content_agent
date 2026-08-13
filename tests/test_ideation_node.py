from content_agent.nodes.ideation import _AngleSelection, ideation_agent
from content_agent.state import AgentState


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "request": {"client_id": "acme", "channel": "whatsapp", "campaign_topic": "end of season sale"},
        "brand_guidelines": None,
        "brief": None,
        "angle": None,
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
    assert result["angle"] == "stub angle"
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
            return _AngleSelection(angle="whatever")

    monkeypatch.setattr("content_agent.nodes.ideation._structured_llm", _RecordingAngleLLM())

    ideation_agent(_base_state(human_edit_notes="make it more urgent"))

    assert "make it more urgent" in captured["prompt"]
    assert "previous angle was rejected" in captured["prompt"]


def test_ideation_agent_no_edit_notes_omits_rejection_language(monkeypatch):
    captured = {}

    class _RecordingAngleLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return _AngleSelection(angle="whatever")

    monkeypatch.setattr("content_agent.nodes.ideation._structured_llm", _RecordingAngleLLM())

    ideation_agent(_base_state())

    assert "previous angle was rejected" not in captured["prompt"]
