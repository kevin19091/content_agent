from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from content_agent.state import AgentState
from content_agent.tools.brand_guidelines import brand_guidelines_tool
from content_agent.tools.brief import brief_creation_tool

load_dotenv()


class _AngleSelection(BaseModel):
    angle: str = Field(description="the single chosen content angle/hook for this campaign")


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
_structured_llm = _llm.with_structured_output(_AngleSelection)


def ideation_agent(state: AgentState) -> dict:
    """PRD §6.1. Fetches guidelines + brief via tools (brief_creation_tool
    already derives key_message/target_audience/constraints/cta -- this
    node doesn't re-derive them), then picks exactly ONE angle. Planning
    only, no copy drafting -- that's content_creation_agent's job."""
    request = state["request"]
    notes = state.get("human_edit_notes")

    guidelines = brand_guidelines_tool(request["client_id"], request["channel"])
    brief = brief_creation_tool(request["client_id"], request["channel"], request["campaign_topic"])

    prompt = (
        f"Campaign topic: {request['campaign_topic']}\n"
        f"Channel: {request['channel']}\n\n"
        f"Brand tone rules: {guidelines['tone_rules']}\n"
        f"Prohibited words: {guidelines['prohibited_words']}\n"
        f"Style guide: {guidelines['style_guide']}\n\n"
        f"Brief -- tone: {brief['tone']}, key message: {brief['key_message']}, "
        f"target audience: {brief['target_audience']}, cta: {brief['cta']}, "
        f"constraints: {brief['constraints']}\n\n"
        "Choose exactly one content angle/hook for this campaign. Do not "
        "draft any copy -- just name the angle."
    )
    if notes:
        prompt += f"\n\nThe previous angle was rejected with this feedback: {notes}"

    selection = _structured_llm.invoke(prompt)

    return {
        "angle": selection.angle,
        "brief": brief,
        "brand_guidelines": guidelines,
        "stage": "ideation",
    }
