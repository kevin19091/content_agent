from content_agent.state import AgentState

# PRD §6.4 routing table:
#   stage       | approve ->              | edit ->                 | reject ->
#   ideation    | content_creation_agent  | ideation_agent          | rejected
#   creation    | compliance_agent        | content_creation_agent  | rejected
#   compliance  | approved                | content_creation_agent  | rejected


def route_after_review(state: AgentState) -> str:
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
