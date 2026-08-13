from langgraph.types import interrupt

from content_agent.state import AgentState


def human_review(state: AgentState) -> dict:
    """Single node entered from three points in the graph (PRD §6.4).
    `state["stage"]` -- set by whichever agent just ran -- tells the
    conditional edge after this node which of the three gates it's
    resolving."""
    decision = interrupt(
        {
            "stage": state["stage"],
            "angle": state.get("angle"),
            "brief": state.get("brief"),
            "draft_content": state.get("draft_content"),
            "compliance_result": state.get("compliance_result"),
        }
    )
    return {
        "human_decision": decision["action"],
        "human_edit_notes": decision.get("notes"),
    }
