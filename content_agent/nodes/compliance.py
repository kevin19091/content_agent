from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from content_agent.schemas import ComplianceResult
from content_agent.state import AgentState

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_structured_llm = _llm.with_structured_output(ComplianceResult)


def _channel_rules(channel: str, draft: dict) -> str:
    """Character counts are computed here, in Python, and handed to the
    model as stated facts -- not left for the model to count from raw
    text. Verified live: without this, the model doesn't reliably count
    characters (a genuine LLM weakness, not hallucination) -- e.g. flagged
    a 29-char title as "36 characters, exceeds 65" on a draft nowhere
    near the limit. Comparing two given numbers is a task the model
    handles fine; counting characters in text is not."""
    if channel == "whatsapp":
        body_len = len(draft.get("body") or "")
        status = "OVER the limit" if body_len > 1024 else "within the limit"
        return (
            f"WhatsApp rules: body must be at most 1024 characters -- this "
            f"draft's body is exactly {body_len} characters ({status}). If "
            "the template_name/content implies a utility template (e.g. "
            "order updates, OTPs, account alerts), it must not contain "
            "promotional content."
        )
    if channel == "push":
        title_len = len(draft.get("title") or "")
        body_len = len(draft.get("body") or "")
        title_status = "OVER the limit" if title_len > 65 else "within the limit"
        body_status = "OVER the limit" if body_len > 178 else "within the limit"
        return (
            f"Push rules: title must be at most 65 characters -- this "
            f"draft's title is exactly {title_len} characters ({title_status}). "
            f"Body must be at most 178 characters -- this draft's body is "
            f"exactly {body_len} characters ({body_status})."
        )
    return ""


def compliance_agent(state: AgentState) -> dict:
    """PRD §6.3. Single structured-output call folding brand + channel
    checks together (tone match, prohibited word scan, channel rules) --
    no separate rule-based tool functions. This is also the sole enforcer
    of the length limits that used to be pydantic Field constraints on
    WhatsAppContent/PushContent (PRD §5.1, §9). Never routes on its own --
    always hands off to human_review regardless of severity."""
    request = state["request"]
    channel = request["channel"]
    draft = state["draft_content"]
    guidelines = state["brand_guidelines"]

    prompt = (
        f"Review this {channel} draft for brand and channel compliance.\n\n"
        f"Draft: {draft}\n\n"
        f"Brand tone rules: {guidelines['tone_rules']}\n"
        f"Prohibited words (flag any use of these): {guidelines['prohibited_words']}\n"
        f"Style guide: {guidelines['style_guide']}\n\n"
        f"{_channel_rules(channel, draft)}\n\n"
        "List every issue found. Set severity to 'blocking' if any rule is "
        "violated (prohibited word used, over a length limit, wrong "
        "template category), 'minor' for tone/style suggestions that don't "
        "violate a hard rule, or 'none' if there are no issues."
    )

    result = _structured_llm.invoke(prompt)

    return {"compliance_result": result.model_dump(), "stage": "compliance"}
