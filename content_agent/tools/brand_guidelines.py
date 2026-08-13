import json

from content_agent.db import get_connection


def brand_guidelines_tool(client_id: str, channel: str) -> dict:
    """PRD §6.1 -- static, pre-seeded lookup keyed on (client_id, channel).
    Read-only for v1, no authoring path."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT tone_rules, prohibited_words, style_guide "
            "FROM brand_guidelines WHERE client_id = ? AND channel = ?",
            (client_id, channel),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError(f"no brand guidelines for client_id={client_id!r} channel={channel!r}")

    return {
        "tone_rules": json.loads(row["tone_rules"]),
        "prohibited_words": json.loads(row["prohibited_words"]),
        "style_guide": row["style_guide"],
    }
