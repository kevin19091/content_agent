"""Exercises the graph's routing end-to-end -- real node functions, real
classify_decision node, with every LLM call faked via conftest.py's
autouse fixture. Confirms human_review/classify_decision resolve
approve/edit/reject correctly at all three stages (PRD §11.2), and that
collect_request/parse_request resolve free-text intake before ideation
ever runs (PRD §11.1)."""

import itertools

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from content_agent.graph import compile_app

_counter = itertools.count()


def make_app():
    return compile_app(checkpointer=MemorySaver())


def _new_config():
    return {"configurable": {"thread_id": f"t{next(_counter)}"}}


def start(app, config, intake_message="push channel, topic: sale"):
    """Runs the graph from a fresh thread through intake in one shot
    (the fake extraction LLM resolves channel+topic from a single
    well-formed message -- see conftest.py), landing on the first
    ideation interrupt. Dedicated intake tests below drive collect_request
    /parse_request directly instead of using this shortcut."""
    result = app.invoke({"client_id": "acme"}, config=config)
    assert result["__interrupt__"][0].value["stage"] == "intake"
    return app.invoke(Command(resume=intake_message), config=config)


def run(app, decisions):
    """Resolves intake with a canned one-shot message, then drives the
    review loop to completion, resuming with raw text messages from
    `decisions` in order whenever it interrupts. Returns (final_state,
    stages_seen) -- stages_seen only covers the review loop, not intake."""
    config = _new_config()
    result = start(app, config)
    stages_seen = []
    step = 0
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        stages_seen.append(payload["stage"])
        message = decisions[step]
        step += 1
        result = app.invoke(Command(resume=message), config=config)
    assert step == len(decisions), "not all supplied decisions were consumed"
    return result, stages_seen


def test_straight_approve_persists_final_content():
    app = make_app()
    result, stages = run(app, ["approve", "approve", "approve"])
    assert stages == ["ideation", "creation", "compliance"]
    assert result["final_content"] is not None
    assert result["final_content"] == result["draft_content"]


def test_edit_at_ideation_loops_to_ideation_agent():
    app = make_app()
    result, stages = run(app, ["make it punchier", "approve", "approve", "approve"])
    assert stages == ["ideation", "ideation", "creation", "compliance"]
    assert "revised" in result["angle"]
    assert result["final_content"] is not None


def test_edit_at_creation_loops_to_content_creation_agent():
    app = make_app()
    result, stages = run(app, ["approve", "make it punchier", "approve", "approve"])
    assert stages == ["ideation", "creation", "creation", "compliance"]
    assert "revised" in result["draft_content"]["body"]
    assert result["final_content"] is not None


def test_edit_at_compliance_loops_back_and_recompliance_checks():
    app = make_app()
    result, stages = run(app, ["approve", "approve", "make it punchier", "approve", "approve"])
    # edit at compliance sends draft back to content_creation_agent, which
    # re-enters compliance_agent (fixed edge) before re-interrupting.
    assert stages == ["ideation", "creation", "compliance", "creation", "compliance"]
    assert result["final_content"] is not None


def test_human_led_routing_edit_at_compliance_reaches_ideation_directly():
    """PRD §11.3 -- the gap human-led routing closes: v1 always forced
    edit-at-compliance through content_creation_agent, even when the
    human wanted the angle redone. "angle" is the fake decision LLM's
    (conftest.py) keyword for target_stage=ideation."""
    app = make_app()
    result, stages = run(
        app, ["approve", "approve", "go back and change the angle", "approve", "approve", "approve"]
    )
    assert stages == ["ideation", "creation", "compliance", "ideation", "creation", "compliance"]
    assert result["final_content"] is not None


def test_ideation_offers_three_angles_and_a_non_default_pick_is_honored():
    """PRD §11.5 -- three candidates are proposed; picking a non-default
    one is data on the approve action, not a fresh ideation_agent call
    (angle changes but angle_options doesn't -- same three options)."""
    app = make_app()
    config = _new_config()

    result = start(app, config)
    payload = result["__interrupt__"][0].value
    assert payload["angle_options"] == ["stub angle", "stub angle B", "stub angle C"]
    assert payload["angle"] == "stub angle"  # the recommendation, angle_options[0]

    result = app.invoke(Command(resume="let's go with the second one, approve"), config=config)
    state = app.get_state(config).values
    assert state["angle"] == "stub angle B"  # overridden to the pick, not the recommendation
    assert state["angle_options"] == ["stub angle", "stub angle B", "stub angle C"]  # unchanged
    assert result["__interrupt__"][0].value["stage"] == "creation"  # moved forward normally


@pytest.mark.parametrize(
    "decisions,expected_stages",
    [
        (["reject this"], ["ideation"]),
        (["approve", "reject this"], ["ideation", "creation"]),
        (["approve", "approve", "reject this"], ["ideation", "creation", "compliance"]),
    ],
)
def test_reject_ends_immediately_with_no_final_content(decisions, expected_stages):
    app = make_app()
    result, stages = run(app, decisions)
    assert stages == expected_stages
    assert result["final_content"] is None


def test_multiple_edits_in_a_row_at_same_stage():
    app = make_app()
    result, stages = run(app, ["make it punchier", "try again", "approve", "approve", "approve"])
    assert stages == ["ideation", "ideation", "ideation", "creation", "compliance"]
    assert result["final_content"] is not None


# --- intake (PRD §11.1) ---------------------------------------------------


def test_intake_accumulates_across_turns_instead_of_re_asking():
    app = make_app()
    config = _new_config()

    result = app.invoke({"client_id": "acme"}, config=config)
    assert result["__interrupt__"][0].value["stage"] == "intake"

    # topic only, first turn
    result = app.invoke(Command(resume="topic: end of season sale"), config=config)
    payload = result["__interrupt__"][0].value
    assert payload["stage"] == "intake"
    assert payload["pending_campaign_topic"] == "end of season sale"
    assert payload["pending_channel"] is None  # still missing, asked again

    # channel only, second turn -- topic from turn 1 must not be lost
    result = app.invoke(Command(resume="whatsapp"), config=config)
    assert result["__interrupt__"][0].value["stage"] == "ideation"  # resolved, moved on
    state = app.get_state(config).values
    assert state["request"] == {
        "client_id": "acme",
        "channel": "whatsapp",
        "campaign_topic": "end of season sale",
    }


def test_intake_resolves_in_one_shot_when_both_given_together():
    app = make_app()
    config = _new_config()
    app.invoke({"client_id": "acme"}, config=config)
    result = app.invoke(Command(resume="push channel, topic: flash sale"), config=config)
    assert result["__interrupt__"][0].value["stage"] == "ideation"


def test_intake_cancel_ends_run_with_no_final_content():
    app = make_app()
    config = _new_config()
    app.invoke({"client_id": "acme"}, config=config)
    result = app.invoke(Command(resume="actually never mind, cancel this"), config=config)
    assert "__interrupt__" not in result
    assert result["final_content"] is None


def test_intake_cancel_after_partial_info_still_ends_cleanly():
    app = make_app()
    config = _new_config()
    app.invoke({"client_id": "acme"}, config=config)
    app.invoke(Command(resume="topic: end of season sale"), config=config)
    result = app.invoke(Command(resume="forget it"), config=config)
    assert "__interrupt__" not in result
    assert result["final_content"] is None


def test_client_id_comes_from_login_never_from_free_text():
    """client_id is set once, at kickoff, from the login username -- never
    extracted from the intake message even if it looks like it could be."""
    app = make_app()
    config = _new_config()
    app.invoke({"client_id": "acme"}, config=config)
    app.invoke(Command(resume="push channel, topic: sale, client_id: someone-else"), config=config)
    state = app.get_state(config).values
    assert state["request"]["client_id"] == "acme"
