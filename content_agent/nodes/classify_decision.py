from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from content_agent.state import AgentState

load_dotenv()


class _DecisionClassification(BaseModel):
    action: Literal["approve", "edit", "reject"]


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_structured_llm = _llm.with_structured_output(_DecisionClassification)

_STAGE_DESCRIPTIONS = {
    "ideation": "the proposed content angle and brief",
    "creation": "a draft of the actual copy",
    "compliance": "a brand/channel compliance check result",
}


def classify_decision(state: AgentState) -> dict:
    """PRD §11.2 -- downstream of human_review, interprets the raw message
    it captured into approve/edit/reject. Moved here from ui.py so it's
    inside _app.invoke() (covered by RetryPolicy and Opik's
    track_langgraph automatically -- no separate @track needed).

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
    action = _structured_llm.invoke(prompt).action

    return {
        "human_decision": action,
        "human_edit_notes": message if action == "edit" else None,
        "node_error": None,
        "node_error_source": None,
    }
