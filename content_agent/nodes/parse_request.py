from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from content_agent.state import AgentState

load_dotenv()


class _RequestExtraction(BaseModel):
    channel: Optional[Literal["whatsapp", "push"]] = None
    campaign_topic: Optional[str] = None
    cancelled: bool = False


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_structured_llm = _llm.with_structured_output(_RequestExtraction)


def parse_request(state: AgentState) -> dict:
    """PRD §11.1 -- extracts channel + campaign_topic from the raw intake
    message. Answers accumulate across turns (pending_channel/
    pending_campaign_topic merged, not overwritten) rather than requiring
    both restated every time route_after_intake loops back to
    collect_request. Also recognizes an explicit cancel -- an uncapped
    ask-again loop needs a way out, or it reads as a trap."""
    message = state["raw_intake"]

    prompt = (
        "A human is starting a new marketing campaign and just sent this "
        f'message:\n\n"{message}"\n\n'
        "Extract, if present:\n"
        "- channel: 'whatsapp' or 'push' -- only if clearly one of these two\n"
        "- campaign_topic: what the campaign is about, in their own words\n\n"
        "If they're explicitly backing out (e.g. \"never mind\", \"cancel\", "
        '"forget it"), set cancelled to true instead.'
    )
    if state.get("pending_channel") or state.get("pending_campaign_topic"):
        prompt += (
            f"\n\nAlready captured from earlier in this conversation -- "
            f"channel: {state.get('pending_channel')!r}, "
            f"topic: {state.get('pending_campaign_topic')!r}. Only extract "
            "NEW information from their latest message above; don't discard "
            "what's already known."
        )

    result = _structured_llm.invoke(prompt)

    if result.cancelled:
        return {"intake_cancelled": True, "node_error": None, "node_error_source": None}

    pending_channel = result.channel or state.get("pending_channel")
    pending_campaign_topic = result.campaign_topic or state.get("pending_campaign_topic")

    updates = {
        "pending_channel": pending_channel,
        "pending_campaign_topic": pending_campaign_topic,
        "node_error": None,
        "node_error_source": None,
    }

    if pending_channel and pending_campaign_topic:
        updates["request"] = {
            "client_id": state["client_id"],
            "channel": pending_channel,
            "campaign_topic": pending_campaign_topic,
        }

    return updates
