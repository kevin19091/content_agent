from content_agent.nodes.classify_decision import _DecisionClassification, classify_decision
from content_agent.state import AgentState


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "request": {"client_id": "acme", "channel": "whatsapp", "campaign_topic": "sale"},
        "brand_guidelines": None,
        "brief": None,
        "angle": None,
        "draft_content": None,
        "compliance_result": None,
        "stage": "ideation",
        "human_message": "approve",
        "human_decision": None,
        "human_edit_notes": None,
        "final_content": None,
        "node_error": None,
        "node_error_source": None,
    }
    state.update(overrides)
    return state


def test_normal_classification_clears_error_fields(monkeypatch):
    monkeypatch.setattr(
        "content_agent.nodes.classify_decision._structured_llm",
        type("F", (), {"invoke": lambda self, p: _DecisionClassification(action="approve")})(),
    )
    result = classify_decision(_base_state())
    assert result["human_decision"] == "approve"
    assert result["node_error"] is None
    assert result["node_error_source"] is None


def test_edit_action_carries_message_as_notes(monkeypatch):
    monkeypatch.setattr(
        "content_agent.nodes.classify_decision._structured_llm",
        type("F", (), {"invoke": lambda self, p: _DecisionClassification(action="edit")})(),
    )
    result = classify_decision(_base_state(human_message="make it punchier"))
    assert result["human_decision"] == "edit"
    assert result["human_edit_notes"] == "make it punchier"


def test_approve_action_has_no_notes(monkeypatch):
    monkeypatch.setattr(
        "content_agent.nodes.classify_decision._structured_llm",
        type("F", (), {"invoke": lambda self, p: _DecisionClassification(action="approve")})(),
    )
    result = classify_decision(_base_state())
    assert result["human_edit_notes"] is None


def test_upstream_node_error_forces_edit_without_calling_llm(monkeypatch):
    """An upstream agent's fields (angle/brief, draft_content,
    compliance_result) were never actually produced -- must not risk an
    LLM reading the reply as 'approve' and routing forward through state
    that doesn't exist (PRD §11.2)."""

    def _should_not_be_called(self, prompt):
        raise AssertionError("LLM must not be called when recovering from an upstream node's failure")

    monkeypatch.setattr(
        "content_agent.nodes.classify_decision._structured_llm",
        type("F", (), {"invoke": _should_not_be_called})(),
    )

    result = classify_decision(
        _base_state(
            human_message="approve",
            node_error="rate limited",
            node_error_source="ideation_agent",
        )
    )
    assert result["human_decision"] == "edit"
    assert result["human_edit_notes"] == "approve"
    assert result["node_error"] is None
    assert result["node_error_source"] is None


def test_upstream_node_error_with_empty_message_still_forces_edit(monkeypatch):
    monkeypatch.setattr(
        "content_agent.nodes.classify_decision._structured_llm",
        type("F", (), {"invoke": lambda self, p: (_ for _ in ()).throw(AssertionError("must not be called"))})(),
    )
    result = classify_decision(
        _base_state(human_message=None, node_error="boom", node_error_source="compliance_agent")
    )
    assert result["human_decision"] == "edit"
    assert result["human_edit_notes"] == "please try again"


def test_classify_decisions_own_prior_failure_does_not_force_edit(monkeypatch):
    """node_error_source == 'classify_decision' means classify_decision's
    OWN previous attempt failed and was recovered -- this fresh attempt
    should classify normally, not force an edit."""
    monkeypatch.setattr(
        "content_agent.nodes.classify_decision._structured_llm",
        type("F", (), {"invoke": lambda self, p: _DecisionClassification(action="approve")})(),
    )
    result = classify_decision(
        _base_state(human_message="approve", node_error="boom", node_error_source="classify_decision")
    )
    assert result["human_decision"] == "approve"
    assert result["node_error"] is None
    assert result["node_error_source"] is None
