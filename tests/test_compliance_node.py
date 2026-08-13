from content_agent.nodes.compliance import compliance_agent
from content_agent.schemas import ComplianceResult
from content_agent.state import AgentState


def _base_state(channel="whatsapp", **overrides) -> AgentState:
    state: AgentState = {
        "request": {"client_id": "acme", "channel": channel, "campaign_topic": "end of season sale"},
        "brand_guidelines": {
            "tone_rules": ["friendly"],
            "prohibited_words": ["guarantee", "free money"],
            "style_guide": "be nice",
        },
        "brief": None,
        "angle": None,
        "draft_content": {"template_name": "t", "body": "Shop now and save!"},
        "compliance_result": None,
        "stage": None,
        "human_decision": None,
        "human_edit_notes": None,
        "final_content": None,
    }
    state.update(overrides)
    return state


def test_returns_compliance_result_shape_and_sets_stage():
    result = compliance_agent(_base_state())
    assert result["stage"] == "compliance"
    assert set(result["compliance_result"].keys()) == {"passed", "issues", "severity"}


def test_never_produces_routing_decision():
    """compliance_agent has no conditional logic of its own -- it always
    hands off to human_review regardless of what it finds (PRD §9)."""
    result = compliance_agent(_base_state())
    assert "human_decision" not in result


def test_prompt_includes_draft_guidelines_and_channel_rules(monkeypatch):
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return ComplianceResult(passed=True, issues=[], severity="none")

    monkeypatch.setattr("content_agent.nodes.compliance._structured_llm", _Recording())
    compliance_agent(_base_state(channel="push"))

    assert "guarantee" in captured["prompt"]
    assert "free money" in captured["prompt"]
    assert "title must be at most 65 characters" in captured["prompt"]


def test_whatsapp_and_push_get_different_channel_rules(monkeypatch):
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return ComplianceResult(passed=True, issues=[], severity="none")

    monkeypatch.setattr("content_agent.nodes.compliance._structured_llm", _Recording())

    compliance_agent(_base_state(channel="whatsapp"))
    whatsapp_prompt = captured["prompt"]

    compliance_agent(_base_state(channel="push"))
    push_prompt = captured["prompt"]

    assert "1024 characters" in whatsapp_prompt
    assert "1024 characters" not in push_prompt
    assert "65 characters" in push_prompt
    assert "65 characters" not in whatsapp_prompt
