from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from content_agent.state import AgentState

load_dotenv()


class _DecisionClassification(BaseModel):
    action: Literal["approve", "edit", "reject"]
    target_stage: Optional[Literal["ideation", "creation"]] = None


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


def classify_decision(state: AgentState) -> dict:
    """PRD §11.2, §11.3 -- downstream of human_review, interprets the raw
    message it captured into approve/edit/reject and, on edit, which stage
    the change belongs at (not always a one-step-back default anymore --
    a human at the compliance gate can send this back to ideation_agent
    directly instead of being forced through content_creation_agent).

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

    result = _structured_llm.invoke(prompt)
    action = result.action

    target_stage = None
    if action == "edit":
        target_stage = "ideation" if stage == "ideation" else (result.target_stage or _DEFAULT_EDIT_TARGET[stage])

    return {
        "human_decision": action,
        "human_edit_notes": message if action == "edit" else None,
        "target_stage": target_stage,
        "node_error": None,
        "node_error_source": None,
    }
