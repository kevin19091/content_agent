from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from content_agent.schemas import PushContent, WhatsAppContent
from content_agent.state import AgentState

load_dotenv()

_CHANNEL_SCHEMAS = {"whatsapp": WhatsAppContent, "push": PushContent}
_CHANNEL_LIMITS = {
    "whatsapp": "body must be at most 1024 characters",
    "push": "title must be at most 65 characters, body must be at most 178 characters",
}

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
_structured_llms = {channel: _llm.with_structured_output(schema) for channel, schema in _CHANNEL_SCHEMAS.items()}


def content_creation_agent(state: AgentState) -> dict:
    """PRD §6.2. Structured output bound to the channel schema. Revision
    context comes from human_edit_notes (edit loop from any gate) or
    compliance_result.issues (only reachable via the compliance gate's
    edit loop, which also carries human_edit_notes -- notes win if both
    are present since they're the more specific, human-authored signal)."""
    request = state["request"]
    channel = request["channel"]
    brief = state["brief"]
    angle = state["angle"]
    brand_guidelines = state["brand_guidelines"]

    notes = state.get("human_edit_notes")
    prior_issues = (state.get("compliance_result") or {}).get("issues")
    prior_draft = state.get("draft_content")

    prompt = (
        f"Write {channel} marketing content for this campaign.\n\n"
        f"Angle: {angle}\n"
        f"Tone: {brief['tone']}\n"
        f"Key message: {brief['key_message']}\n"
        f"Target audience: {brief['target_audience']}\n"
        f"Call to action: {brief['cta']}\n"
        f"Constraints: {brief['constraints']}\n"
        f"Target length: about {brief['word_length']} words\n\n"
        f"Brand style guide: {brand_guidelines['style_guide']}\n"
        f"Prohibited words (never use these): {brand_guidelines['prohibited_words']}\n\n"
        f"Channel limits: {_CHANNEL_LIMITS[channel]}\n"
    )

    if prior_draft and notes:
        prompt += (
            f"\nPrevious draft: {prior_draft}\nHuman edit feedback: {notes}\n"
            "Revise the draft to address this feedback."
        )
    elif prior_draft and prior_issues:
        prompt += (
            f"\nPrevious draft: {prior_draft}\nCompliance issues to fix: {prior_issues}\n"
            "Revise the draft to address these issues."
        )

    draft = _structured_llms[channel].invoke(prompt)

    return {"draft_content": draft.model_dump(), "stage": "creation"}
