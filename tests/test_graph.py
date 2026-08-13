"""Milestone 1: exercise the graph's routing against stub nodes -- no LLM,
no DB, no UI. Confirms the shared human_review node resolves approve/edit/
reject correctly at all three stages before any real agent logic is built
(PRD §10 milestone 1)."""

import itertools

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from content_agent.graph import compile_app

_counter = itertools.count()


def make_app():
    return compile_app(checkpointer=MemorySaver())


def run(app, decisions):
    """Drive the graph to completion, resuming with `decisions` in order
    whenever it interrupts. Returns (final_state, stages_seen)."""
    config = {"configurable": {"thread_id": f"t{next(_counter)}"}}
    result = app.invoke(
        {"request": {"client_id": "acme", "channel": "push", "campaign_topic": "sale"}},
        config=config,
    )
    stages_seen = []
    step = 0
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        stages_seen.append(payload["stage"])
        action = decisions[step]
        step += 1
        resume = {"action": action}
        if action == "edit":
            resume["notes"] = "make it punchier"
        result = app.invoke(Command(resume=resume), config=config)
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
    result, stages = run(app, ["edit", "approve", "approve", "approve"])
    assert stages == ["ideation", "ideation", "creation", "compliance"]
    assert "revised" in result["angle"]
    assert result["final_content"] is not None


def test_edit_at_creation_loops_to_content_creation_agent():
    app = make_app()
    result, stages = run(app, ["approve", "edit", "approve", "approve"])
    assert stages == ["ideation", "creation", "creation", "compliance"]
    assert "revised" in result["draft_content"]["body"]
    assert result["final_content"] is not None


def test_edit_at_compliance_loops_back_and_recompliance_checks():
    app = make_app()
    result, stages = run(app, ["approve", "approve", "edit", "approve", "approve"])
    # edit at compliance sends draft back to content_creation_agent, which
    # re-enters compliance_agent (fixed edge) before re-interrupting.
    assert stages == ["ideation", "creation", "compliance", "creation", "compliance"]
    assert result["final_content"] is not None


@pytest.mark.parametrize(
    "decisions,expected_stages",
    [
        (["reject"], ["ideation"]),
        (["approve", "reject"], ["ideation", "creation"]),
        (["approve", "approve", "reject"], ["ideation", "creation", "compliance"]),
    ],
)
def test_reject_ends_immediately_with_no_final_content(decisions, expected_stages):
    app = make_app()
    result, stages = run(app, decisions)
    assert stages == expected_stages
    assert result["final_content"] is None


def test_multiple_edits_in_a_row_at_same_stage():
    app = make_app()
    result, stages = run(app, ["edit", "edit", "approve", "approve", "approve"])
    assert stages == ["ideation", "ideation", "ideation", "creation", "compliance"]
    assert result["final_content"] is not None
