import json

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from content_agent.state import AgentState
from content_agent.tools.brand_guidelines import brand_guidelines_tool
from content_agent.tools.brief import brief_creation_tool

load_dotenv()

_MAX_TOOL_ROUNDS = 4  # 2 tools available -- no legitimate reason to loop past a couple rounds


class _AngleSelection(BaseModel):
    angles: list[str] = Field(
        min_length=3,
        max_length=3,
        description="exactly three candidate content angles/hooks for this campaign, ordered strongest first",
    )


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
_structured_llm = _llm.with_structured_output(_AngleSelection)

_tool_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _build_tools(request: dict) -> list:
    """No-arg tools -- client_id/channel/campaign_topic are already known
    from state, so the model only ever decides *whether* to call these, not
    what to pass them. Keeps it from ever being able to fetch another
    client's data by mis-supplying an argument."""

    @tool
    def fetch_brand_guidelines() -> dict:
        """Fetch this client's brand tone rules, prohibited words, and style guide for this channel."""
        return brand_guidelines_tool(request["client_id"], request["channel"])

    @tool
    def fetch_brief() -> dict:
        """Derive the campaign brief (tone, word length, key message, target audience, constraints, CTA) for this topic."""
        return brief_creation_tool(request["client_id"], request["channel"], request["campaign_topic"])

    return [fetch_brand_guidelines, fetch_brief]


def _resolve_guidelines_and_brief(request: dict, existing_guidelines: dict | None, existing_brief: dict | None):
    """PRD §11.6 -- `.bind_tools()` + a small loop so the model decides
    whether it needs either tool at all, instead of always calling both
    unconditionally. On a re-run after an edit (same campaign_topic),
    it can reasonably decide guidelines/brief already in state are still
    current and skip re-fetching.

    Falls back to fetching directly if nothing exists yet and the model
    didn't call the tool anyway -- this node's output shouldn't depend on
    the model reliably remembering to call a tool for data it has no
    substitute for."""
    tools = _build_tools(request)
    tools_by_name = {t.name: t for t in tools}
    bound_llm = _tool_llm.bind_tools(tools)

    known = []
    if existing_guidelines is not None:
        known.append(f"Brand guidelines already fetched this run: {existing_guidelines}")
    if existing_brief is not None:
        known.append(f"Brief already created this run: {existing_brief}")
    known_block = "\n".join(known) if known else "Nothing has been fetched yet this run."

    messages = [
        SystemMessage(
            "You are gathering inputs for a content ideation step. Two tools are "
            "available: fetch_brand_guidelines and fetch_brief. Call whichever of "
            "them you still need fresh data for. If what you already have below is "
            "still current for this exact campaign topic and channel, don't call "
            "the corresponding tool again."
        ),
        HumanMessage(f"Campaign topic: {request['campaign_topic']}\nChannel: {request['channel']}\n\n{known_block}"),
    ]

    guidelines, brief = existing_guidelines, existing_brief
    for _ in range(_MAX_TOOL_ROUNDS):
        response = bound_llm.invoke(messages)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        messages.append(response)
        for call in tool_calls:
            result = tools_by_name[call["name"]].invoke(call["args"])
            if call["name"] == "fetch_brand_guidelines":
                guidelines = result
            elif call["name"] == "fetch_brief":
                brief = result
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))

    if guidelines is None:
        guidelines = brand_guidelines_tool(request["client_id"], request["channel"])
    if brief is None:
        brief = brief_creation_tool(request["client_id"], request["channel"], request["campaign_topic"])

    return guidelines, brief


def ideation_agent(state: AgentState) -> dict:
    """PRD §6.1, §11.5, §11.6. Fetches guidelines + brief via optional tool
    calls (brief_creation_tool already derives key_message/target_audience/
    constraints/cta -- this node doesn't re-derive them), then proposes
    three candidate angles, strongest first. Planning only, no copy
    drafting -- that's content_creation_agent's job.

    `angle` (the one actually drafted from) defaults to the recommendation
    (angle_options[0]); classify_decision can override it if the human
    picks a different option (see selected_angle_index there)."""
    request = state["request"]
    notes = state.get("human_edit_notes")

    guidelines, brief = _resolve_guidelines_and_brief(request, state.get("brand_guidelines"), state.get("brief"))

    prompt = (
        f"Campaign topic: {request['campaign_topic']}\n"
        f"Channel: {request['channel']}\n\n"
        f"Brand tone rules: {guidelines['tone_rules']}\n"
        f"Prohibited words: {guidelines['prohibited_words']}\n"
        f"Style guide: {guidelines['style_guide']}\n\n"
        f"Brief -- tone: {brief['tone']}, key message: {brief['key_message']}, "
        f"target audience: {brief['target_audience']}, cta: {brief['cta']}, "
        f"constraints: {brief['constraints']}\n\n"
        "Propose exactly three distinct content angles/hooks for this "
        "campaign, ordered with the strongest one first. Do not draft any "
        "copy -- just name the angles."
    )
    if notes:
        prompt += f"\n\nThe previous angle(s) needed rework, per this feedback: {notes}"

    selection = _structured_llm.invoke(prompt)

    return {
        "angle": selection.angles[0],
        "angle_options": selection.angles,
        "brief": brief,
        "brand_guidelines": guidelines,
        "stage": "ideation",
    }
