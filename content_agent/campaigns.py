import json
from datetime import datetime, timezone
from typing import Optional

from content_agent.db import get_connection


def save_campaign(
    thread_id: str,
    client_id: str,
    channel: Optional[str],
    campaign_topic: Optional[str],
    status: str,
    chat_history: list,
) -> None:
    """PRD §11.7 -- upsert, not derived from LangGraph's own checkpoint
    blobs (those are keyed for resuming one thread_id, not for listing
    across a client's campaigns). chat_history is stored as the exact
    formatted transcript, not re-derived from graph state on load, so a
    resumed campaign shows precisely what the human saw live."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO campaigns
                (thread_id, client_id, channel, campaign_topic, status, chat_history, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                client_id = excluded.client_id,
                channel = excluded.channel,
                campaign_topic = excluded.campaign_topic,
                status = excluded.status,
                chat_history = excluded.chat_history,
                updated_at = excluded.updated_at
            """,
            (thread_id, client_id, channel, campaign_topic, status, json.dumps(chat_history), now, now),
        )
        conn.commit()
    finally:
        conn.close()


def list_campaigns(client_id: str) -> list[dict]:
    """Most recently updated first -- scoped to one client, per PRD §11.7."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT thread_id, channel, campaign_topic, status, updated_at "
            "FROM campaigns WHERE client_id = ? ORDER BY updated_at DESC",
            (client_id,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def get_campaign(thread_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM campaigns WHERE thread_id = ?", (thread_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    campaign = dict(row)
    campaign["chat_history"] = json.loads(campaign["chat_history"])
    return campaign
