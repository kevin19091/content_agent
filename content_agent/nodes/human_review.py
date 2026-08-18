from langgraph.types import interrupt

from content_agent.state import AgentState


def human_review(state: AgentState) -> dict:
    """PRD §11.2 -- interrupt-only, entered from three points in the graph
    plus as classify_decision's re-prompt target. Just pauses and captures
    the raw message; interpretation happens downstream in
    classify_decision. Has to stay this way -- a node containing
    interrupt() can't safely carry a RetryPolicy (confirmed by testing:
    retrying it doesn't re-run the code after interrupt() resolves, it
    collapses back to re-presenting the same interrupt)."""
    message = interrupt(
        {
            "stage": state["stage"],
            "angle": state.get("angle"),
            "angle_options": state.get("angle_options"),
            "brief": state.get("brief"),
            "draft_content": state.get("draft_content"),
            "compliance_result": state.get("compliance_result"),
            "node_error": state.get("node_error"),
        }
    )
    return {"human_message": message}
