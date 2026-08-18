from langgraph.types import interrupt

from content_agent.state import AgentState


def collect_request(state: AgentState) -> dict:
    """PRD §11.1 -- interrupt-only entry point, pauses and captures free
    text: either the initial campaign kickoff message, or a follow-up
    answer during intake. Interpretation happens downstream in
    parse_request -- same split as human_review/classify_decision (§11.2):
    a node containing interrupt() can't safely carry a RetryPolicy."""
    message = interrupt(
        {
            "stage": "intake",
            "pending_channel": state.get("pending_channel"),
            "pending_campaign_topic": state.get("pending_campaign_topic"),
            "node_error": state.get("node_error"),
        }
    )
    return {"raw_intake": message}
