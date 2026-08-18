from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from content_agent.state import AgentState

load_dotenv()


class _DecisionClassification(BaseModel):
    action: Literal["approve", "edit", "reject"]
    target_stage: Optional[Literal["ideation", "creation"]] = None
    selected_angle_index: Optional[int] = None


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_structured_llm = _llm.with_structured_output(_DecisionClassification)

_STAGE_DESCRIPTIONS = {
    "ideation": "the proposed content angle and brief",
    "creation": "a draft of the actual copy",
    "compliance": "a brand/channel compliance check result",
}

# PRD §11.3 -- human-led routing. When a human at any stage asks for a
# change without naming where it should happen, fall back to this: the
# same one-step-back target the old fixed routing table used. Only
# "creation" and "compliance" ever have a genuine second option (going
# all the way back to the angle); "ideation" has nowhere else to go, so
# it's never even put to the LLM (see classify_decision below).
_DEFAULT_EDIT_TARGET = {
    "ideation": "ideation",
    "creation": "creation",
    "compliance": "creation",
}

_TARGET_STAGE_INSTRUCTIONS = (
    "\n\nIf the action is 'edit', also decide which stage the change belongs "
    "at:\n"
    "- 'ideation': they want a different angle/hook, or a brief-level change "
    '(e.g. "go back and change the angle", "let\'s rethink this entirely")\n'
    "- 'creation': they want the copy itself revised without changing the "
    'underlying angle (e.g. "make it punchier", "shorten the body")\n'
    "If it's not clear which they mean, default to '{default}'."
)

_ANGLE_SELECTION_INSTRUCTIONS = (
    "\n\nThree candidate angles were shown, in this order (#1 is the "
    "recommendation):\n{options}\n\n"
    "If the action is 'approve' and they specifically named a different "
    'option than the recommendation (e.g. "let\'s go with the second one", '
    '"I like option 3 best"), set selected_angle_index to that option\'s '
    "position, 0-based (so option 2 -> 1, option 3 -> 2). Otherwise leave "
    "it unset -- the recommendation is used by default."
)


def classify_decision(state: AgentState) -> dict:
    """PRD §11.2, §11.3, §11.5 -- downstream of human_review, interprets
    the raw message it captured into approve/edit/reject, on edit which
    stage the change belongs at, and on approve at stage=ideation which of
    the three proposed angles was actually meant (defaults to the
    recommendation, angle_options[0], if unspecified).

    If node_error was set by an upstream AGENT's exhausted retries (not by
    classify_decision's own -- see node_error_source), that agent's fields
    (angle/brief, draft_content, compliance_result) were never actually
    produced. Skip classification and force a retry of that step instead
    of risking an LLM reading the reply as "approve" and routing forward
    through state that doesn't exist."""
    if state.get("node_error") and state.get("node_error_source") != "classify_decision":
        return {
            "human_decision": "edit",
            "human_edit_notes": state.get("human_message") or "please try again",
            "target_stage": _DEFAULT_EDIT_TARGET[state["stage"]],
            "node_error": None,
            "node_error_source": None,
        }

    stage = state["stage"]
    message = state["human_message"]
    angle_options = state.get("angle_options") or []

    prompt = (
        f"A human is reviewing {_STAGE_DESCRIPTIONS.get(stage, 'an agent output')} "
        f"in a content approval workflow and just sent this message:\n\n\"{message}\"\n\n"
        "Classify their intent as exactly one of:\n"
        "- approve: satisfied, move on as-is\n"
        "- edit: wants changes; their message is the feedback to act on\n"
        "- reject: wants to kill this campaign entirely, not just request changes"
    )
    if stage != "ideation":
        prompt += _TARGET_STAGE_INSTRUCTIONS.format(default=_DEFAULT_EDIT_TARGET[stage])
    if stage == "ideation" and angle_options:
        options_text = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(angle_options))
        prompt += _ANGLE_SELECTION_INSTRUCTIONS.format(options=options_text)

    result = _structured_llm.invoke(prompt)
    action = result.action

    target_stage = None
    if action == "edit":
        target_stage = "ideation" if stage == "ideation" else (result.target_stage or _DEFAULT_EDIT_TARGET[stage])

    updates = {
        "human_decision": action,
        "human_edit_notes": message if action == "edit" else None,
        "target_stage": target_stage,
        "node_error": None,
        "node_error_source": None,
    }

    if (
        stage == "ideation"
        and action == "approve"
        and result.selected_angle_index is not None
        and 0 <= result.selected_angle_index < len(angle_options)
    ):
        updates["angle"] = angle_options[result.selected_angle_index]

    return updates
