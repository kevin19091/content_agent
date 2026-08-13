from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from content_agent.db import get_connection

load_dotenv()


class _DerivedBriefFields(BaseModel):
    key_message: str = Field(description="the single core message the content should communicate")
    target_audience: str = Field(description="who this campaign is speaking to")
    constraints: list[str] = Field(description="content constraints implied by the campaign topic")
    cta: str = Field(description="the call to action")


_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
_structured_llm = _llm.with_structured_output(_DerivedBriefFields)


def _lookup_tone_and_word_length(client_id: str, channel: str) -> dict:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT tone, word_length FROM client_channel_brief WHERE client_id = ? AND channel = ?",
            (client_id, channel),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError(f"no brief config for client_id={client_id!r} channel={channel!r}")

    return {"tone": row["tone"], "word_length": row["word_length"]}


def brief_creation_tool(client_id: str, channel: str, campaign_topic: str) -> dict:
    """PRD §6.1, §9 -- hybrid, split by field. `tone`/`word_length` are
    always a DB lookup keyed on (client_id, channel); the rest are always
    LLM-derived from campaign_topic. DB values win over anything the LLM
    proposes for tone/word_length (it isn't asked to propose them)."""
    db_fields = _lookup_tone_and_word_length(client_id, channel)

    derived = _structured_llm.invoke(
        f"Campaign topic: {campaign_topic}\n"
        f"Channel: {channel}\n"
        f"Tone to write in: {db_fields['tone']}\n\n"
        "Derive the key message, target audience, constraints, and call to "
        "action for this campaign."
    )

    return {
        "tone": db_fields["tone"],
        "word_length": db_fields["word_length"],
        "key_message": derived.key_message,
        "target_audience": derived.target_audience,
        "constraints": derived.constraints,
        "cta": derived.cta,
    }
