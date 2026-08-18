from content_agent.routing import route_after_review


def _state(**overrides):
    base = {
        "stage": "ideation",
        "human_decision": "approve",
        "target_stage": None,
        "node_error": None,
    }
    base.update(overrides)
    return base


def test_node_error_routes_to_human_review_regardless_of_decision():
    """node_error still set here means classify_decision's own body never
    ran (bypassed by update_state during recovery) -- re-prompt, don't try
    to interpret a human_decision that was never actually produced."""
    result = route_after_review(_state(node_error="boom", human_decision=None, stage=None))
    assert result == "human_review"


def test_no_node_error_uses_normal_stage_decision_table():
    assert route_after_review(_state(stage="ideation", human_decision="approve")) == "content_creation_agent"
    assert route_after_review(_state(stage="creation", human_decision="approve")) == "compliance_agent"
    assert route_after_review(_state(stage="compliance", human_decision="approve")) == "approved"
    assert route_after_review(_state(stage="ideation", human_decision="reject")) == "rejected"


def test_edit_routes_to_target_stage_set_by_classify_decision():
    """PRD §11.3 -- edit's destination is human-led now, not a fixed
    one-step-back: it's whatever classify_decision put in target_stage."""
    assert (
        route_after_review(_state(stage="ideation", human_decision="edit", target_stage="ideation"))
        == "ideation_agent"
    )
    assert (
        route_after_review(_state(stage="creation", human_decision="edit", target_stage="creation"))
        == "content_creation_agent"
    )
    # the gap this feature closes: edit at compliance can reach ideation_agent
    # directly now, not just content_creation_agent
    assert (
        route_after_review(_state(stage="compliance", human_decision="edit", target_stage="ideation"))
        == "ideation_agent"
    )
    assert (
        route_after_review(_state(stage="compliance", human_decision="edit", target_stage="creation"))
        == "content_creation_agent"
    )


def test_edit_with_missing_target_stage_falls_back_to_content_creation_agent():
    """Defensive default if target_stage is ever absent -- shouldn't
    happen in practice (classify_decision always sets it on edit), but
    routing must not crash if it does."""
    assert (
        route_after_review(_state(stage="creation", human_decision="edit", target_stage=None))
        == "content_creation_agent"
    )


def test_falsy_node_error_does_not_trigger_reprompt():
    """node_error="" or missing should behave like None, not trip the
    re-prompt branch."""
    assert route_after_review(_state(node_error="", stage="ideation", human_decision="approve")) == "content_creation_agent"
