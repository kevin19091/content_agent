from content_agent.state import AgentState

# PRD §6.4/§11.2 routing table (after classify_decision):
#   node_error set (classify_decision's own retries exhausted) -> human_review, re-prompt
#   stage       | approve ->              | edit ->                 | reject ->
#   ideation    | content_creation_agent  | ideation_agent          | rejected
#   creation    | compliance_agent        | content_creation_agent  | rejected
#   compliance  | approved                | content_creation_agent  | rejected


def route_after_review(state: AgentState) -> str:
    if state.get("node_error"):
        # classify_decision's own body never ran -- it was bypassed by
        # update_state(as_node="classify_decision", ...) during recovery.
        # If its body HAD run (guard path or normal path), it always
        # clears node_error, so reaching here with it still set means only
        # one thing: re-prompt.
        return "human_review"

    stage = state["stage"]
    decision = state["human_decision"]

    if decision == "reject":
        return "rejected"

    if decision == "edit":
        return "ideation_agent" if stage == "ideation" else "content_creation_agent"

    if decision == "approve":
        if stage == "ideation":
            return "content_creation_agent"
        if stage == "creation":
            return "compliance_agent"
        if stage == "compliance":
            return "approved"

    raise ValueError(f"unhandled (stage={stage!r}, decision={decision!r})")
