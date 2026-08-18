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
        "target_stage": None,
        "final_content": None,
        "node_error": None,
        "node_error_source": None,
    }
    state.update(overrides)
    return state


def _fake_llm(action, target_stage=None):
    return type("F", (), {"invoke": lambda self, p: _DecisionClassification(action=action, target_stage=target_stage)})()


def test_normal_classification_clears_error_fields(monkeypatch):
    monkeypatch.setattr("content_agent.nodes.classify_decision._structured_llm", _fake_llm("approve"))
    result = classify_decision(_base_state())
    assert result["human_decision"] == "approve"
    assert result["node_error"] is None
    assert result["node_error_source"] is None


def test_edit_action_carries_message_as_notes(monkeypatch):
    monkeypatch.setattr("content_agent.nodes.classify_decision._structured_llm", _fake_llm("edit"))
    result = classify_decision(_base_state(human_message="make it punchier"))
    assert result["human_decision"] == "edit"
    assert result["human_edit_notes"] == "make it punchier"


def test_approve_action_has_no_notes(monkeypatch):
    monkeypatch.setattr("content_agent.nodes.classify_decision._structured_llm", _fake_llm("approve"))
    result = classify_decision(_base_state())
    assert result["human_edit_notes"] is None
    assert result["target_stage"] is None


def test_approve_and_reject_never_set_a_target_stage(monkeypatch):
    monkeypatch.setattr("content_agent.nodes.classify_decision._structured_llm", _fake_llm("reject"))
    result = classify_decision(_base_state(stage="compliance"))
    assert result["target_stage"] is None


def test_edit_at_ideation_always_targets_ideation_even_if_llm_says_otherwise(monkeypatch):
    """PRD §11.3 -- there's nowhere else to go from ideation, so the LLM
    isn't even asked; target_stage is hard-coded regardless of what a
    misbehaving structured output might return."""
    monkeypatch.setattr("content_agent.nodes.classify_decision._structured_llm", _fake_llm("edit", target_stage="creation"))
    result = classify_decision(_base_state(stage="ideation"))
    assert result["target_stage"] == "ideation"


def test_edit_at_creation_honors_explicit_ideation_target(monkeypatch):
    monkeypatch.setattr("content_agent.nodes.classify_decision._structured_llm", _fake_llm("edit", target_stage="ideation"))
    result = classify_decision(_base_state(stage="creation", human_message="let's rethink the angle"))
    assert result["target_stage"] == "ideation"


def test_edit_at_creation_defaults_to_creation_when_llm_gives_no_target(monkeypatch):
    monkeypatch.setattr("content_agent.nodes.classify_decision._structured_llm", _fake_llm("edit", target_stage=None))
    result = classify_decision(_base_state(stage="creation", human_message="make it punchier"))
    assert result["target_stage"] == "creation"


def test_edit_at_compliance_honors_explicit_ideation_target(monkeypatch):
    """The gap this feature actually closes: v1 forced edit-at-compliance
    to always go through content_creation_agent, even when the human
    wanted the angle redone."""
    monkeypatch.setattr("content_agent.nodes.classify_decision._structured_llm", _fake_llm("edit", target_stage="ideation"))
    result = classify_decision(_base_state(stage="compliance", human_message="go back and change the angle"))
    assert result["target_stage"] == "ideation"


def test_edit_at_compliance_defaults_to_creation_when_llm_gives_no_target(monkeypatch):
    monkeypatch.setattr("content_agent.nodes.classify_decision._structured_llm", _fake_llm("edit", target_stage=None))
    result = classify_decision(_base_state(stage="compliance", human_message="make the CTA stronger"))
    assert result["target_stage"] == "creation"


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
            stage="ideation",
            human_message="approve",
            node_error="rate limited",
            node_error_source="ideation_agent",
        )
    )
    assert result["human_decision"] == "edit"
    assert result["human_edit_notes"] == "approve"
    assert result["target_stage"] == "ideation"
    assert result["node_error"] is None
    assert result["node_error_source"] is None


def test_upstream_node_error_with_empty_message_still_forces_edit(monkeypatch):
    monkeypatch.setattr(
        "content_agent.nodes.classify_decision._structured_llm",
        type("F", (), {"invoke": lambda self, p: (_ for _ in ()).throw(AssertionError("must not be called"))})(),
    )
    result = classify_decision(
        _base_state(stage="compliance", human_message=None, node_error="boom", node_error_source="compliance_agent")
    )
    assert result["human_decision"] == "edit"
    assert result["human_edit_notes"] == "please try again"
    # compliance_agent failing retries via content_creation_agent (its
    # fixed edge re-runs compliance automatically) -- not a direct target.
    assert result["target_stage"] == "creation"


def test_classify_decisions_own_prior_failure_does_not_force_edit(monkeypatch):
    """node_error_source == 'classify_decision' means classify_decision's
    OWN previous attempt failed and was recovered -- this fresh attempt
    should classify normally, not force an edit."""
    monkeypatch.setattr("content_agent.nodes.classify_decision._structured_llm", _fake_llm("approve"))
    result = classify_decision(
        _base_state(human_message="approve", node_error="boom", node_error_source="classify_decision")
    )
    assert result["human_decision"] == "approve"
    assert result["node_error"] is None
    assert result["node_error_source"] is None
