from content_agent.state import AgentState

# PRD §6.4/§11.2/§11.3 routing table (after classify_decision):
#   node_error set (classify_decision's own retries exhausted) -> human_review, re-prompt
#   stage       | approve ->              | edit ->                        | reject ->
#   ideation    | content_creation_agent  | ideation_agent (only option)   | rejected
#   creation    | compliance_agent        | target_stage (ideation/creation) | rejected
#   compliance  | approved                | target_stage (ideation/creation) | rejected
#
# edit's target is human-led, not a fixed one-step-back -- classify_decision
# sets target_stage from what the human actually asked for (defaulting to
# the old one-step-back behavior when they didn't specify).


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
        target_stage = state.get("target_stage")
        return "ideation_agent" if target_stage == "ideation" else "content_creation_agent"

    if decision == "approve":
        if stage == "ideation":
            return "content_creation_agent"
        if stage == "creation":
            return "compliance_agent"
        if stage == "compliance":
            return "approved"

    raise ValueError(f"unhandled (stage={stage!r}, decision={decision!r})")


def route_after_intake(state: AgentState) -> str:
    """PRD §11.1 -- parse_request's conditional edge. No cap on the
    ask-again loop: missing info is a normal conversation, not a failure
    (RetryPolicy covers the separate case of parse_request's own LLM call
    actually erroring)."""
    if state.get("node_error"):
        # parse_request's own body never ran -- bypassed by
        # update_state(as_node="parse_request", ...) during recovery.
        return "collect_request"

    if state.get("intake_cancelled"):
        return "rejected"

    if state.get("pending_channel") and state.get("pending_campaign_topic"):
        return "ideation_agent"

    return "collect_request"
