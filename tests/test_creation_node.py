from content_agent.nodes.creation import content_creation_agent
from content_agent.schemas import PushContent, WhatsAppContent
from content_agent.state import AgentState


def _base_state(channel="whatsapp", **overrides) -> AgentState:
    state: AgentState = {
        "request": {"client_id": "acme", "channel": channel, "campaign_topic": "end of season sale"},
        "brand_guidelines": {"tone_rules": [], "prohibited_words": ["guarantee"], "style_guide": "be nice"},
        "brief": {
            "tone": "warm",
            "key_message": "big savings",
            "target_audience": "everyone",
            "constraints": [],
            "cta": "shop now",
            "word_length": 40,
        },
        "angle": "seasonal savings",
        "draft_content": None,
        "compliance_result": None,
        "stage": None,
        "human_decision": None,
        "human_edit_notes": None,
        "final_content": None,
    }
    state.update(overrides)
    return state


def test_selects_whatsapp_schema():
    result = content_creation_agent(_base_state(channel="whatsapp"))
    assert result["stage"] == "creation"
    assert set(result["draft_content"].keys()) == set(WhatsAppContent.model_fields.keys())


def test_selects_push_schema():
    result = content_creation_agent(_base_state(channel="push"))
    assert set(result["draft_content"].keys()) == set(PushContent.model_fields.keys())


def test_no_prior_draft_produces_fresh_content(monkeypatch):
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return WhatsAppContent(template_name="t", body="fresh")

    monkeypatch.setattr("content_agent.nodes.creation._structured_llms", {"whatsapp": _Recording(), "push": _Recording()})
    content_creation_agent(_base_state(channel="whatsapp"))

    assert "Revise the draft" not in captured["prompt"]


def test_human_edit_notes_trigger_revision_with_prior_draft(monkeypatch):
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return WhatsAppContent(template_name="t", body="revised")

    monkeypatch.setattr("content_agent.nodes.creation._structured_llms", {"whatsapp": _Recording(), "push": _Recording()})

    prior = {"template_name": "t", "body": "old body"}
    content_creation_agent(
        _base_state(channel="whatsapp", draft_content=prior, human_edit_notes="make it punchier")
    )

    assert "make it punchier" in captured["prompt"]
    assert "old body" in captured["prompt"]
    assert "Revise the draft" in captured["prompt"]


def test_compliance_issues_trigger_revision_when_no_human_notes(monkeypatch):
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return WhatsAppContent(template_name="t", body="revised")

    monkeypatch.setattr("content_agent.nodes.creation._structured_llms", {"whatsapp": _Recording(), "push": _Recording()})

    prior = {"template_name": "t", "body": "old body"}
    content_creation_agent(
        _base_state(
            channel="whatsapp",
            draft_content=prior,
            compliance_result={"passed": False, "issues": ["body too long"], "severity": "blocking"},
        )
    )

    assert "body too long" in captured["prompt"]
    assert "Revise the draft" in captured["prompt"]


def test_human_notes_win_over_compliance_issues_when_both_present(monkeypatch):
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return WhatsAppContent(template_name="t", body="revised")

    monkeypatch.setattr("content_agent.nodes.creation._structured_llms", {"whatsapp": _Recording(), "push": _Recording()})

    prior = {"template_name": "t", "body": "old body"}
    content_creation_agent(
        _base_state(
            channel="whatsapp",
            draft_content=prior,
            human_edit_notes="say it differently",
            compliance_result={"passed": False, "issues": ["body too long"], "severity": "blocking"},
        )
    )

    assert "say it differently" in captured["prompt"]
    assert "body too long" not in captured["prompt"]
