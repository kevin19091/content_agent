from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from content_agent.schemas import ComplianceResult
from content_agent.state import AgentState

load_dotenv()

_CHANNEL_RULES = {
    "whatsapp": (
        "WhatsApp rules: body must be at most 1024 characters. If the "
        "template_name/content implies a utility template (e.g. order "
        "updates, OTPs, account alerts), it must not contain promotional "
        "content."
    ),
    "push": (
        "Push rules: title must be at most 65 characters, body must be at "
        "most 178 characters."
    ),
}

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_structured_llm = _llm.with_structured_output(ComplianceResult)


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
        f"{_CHANNEL_RULES[channel]}\n\n"
        "List every issue found. Set severity to 'blocking' if any rule is "
        "violated (prohibited word used, over a length limit, wrong "
        "template category), 'minor' for tone/style suggestions that don't "
        "violate a hard rule, or 'none' if there are no issues."
    )

    result = _structured_llm.invoke(prompt)

    return {"compliance_result": result.model_dump(), "stage": "compliance"}
