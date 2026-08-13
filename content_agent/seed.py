"""Seed static per-(client_id, channel) brand guidelines + brief config.
Static/pre-seeded for v1 -- no authoring UI (PRD §7, §9)."""

from content_agent.db import get_connection, init_db

BRAND_GUIDELINES = [
    {
        "client_id": "acme",
        "channel": "whatsapp",
        "tone_rules": ["friendly", "concise", "no exclamation spam"],
        "prohibited_words": ["guarantee", "free money", "act now"],
        "style_guide": 'Use first-person plural ("we"); always include a clear CTA; avoid ALL CAPS.',
    },
    {
        "client_id": "acme",
        "channel": "push",
        "tone_rules": ["punchy", "urgent but not pushy"],
        "prohibited_words": ["guarantee", "free money", "act now"],
        "style_guide": "Lead with the benefit in the title; keep body to one short sentence.",
    },
]

CLIENT_CHANNEL_BRIEF = [
    {"client_id": "acme", "channel": "whatsapp", "tone": "warm and conversational", "word_length": 60},
    {"client_id": "acme", "channel": "push", "tone": "punchy and urgent", "word_length": 20},
]


def seed() -> None:
    import json

    init_db()
    conn = get_connection()
    try:
        for row in BRAND_GUIDELINES:
            conn.execute(
                """INSERT OR REPLACE INTO brand_guidelines
                   (client_id, channel, tone_rules, prohibited_words, style_guide)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    row["client_id"],
                    row["channel"],
                    json.dumps(row["tone_rules"]),
                    json.dumps(row["prohibited_words"]),
                    row["style_guide"],
                ),
            )
        for row in CLIENT_CHANNEL_BRIEF:
            conn.execute(
                """INSERT OR REPLACE INTO client_channel_brief
                   (client_id, channel, tone, word_length)
                   VALUES (?, ?, ?, ?)""",
                (row["client_id"], row["channel"], row["tone"], row["word_length"]),
            )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    seed()
    print("seeded.")
