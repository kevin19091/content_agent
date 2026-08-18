import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.environ.get("CONTENT_AGENT_DB_PATH", "content_agent.db")).resolve()

SCHEMA = """
CREATE TABLE IF NOT EXISTS brand_guidelines (
    client_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    tone_rules TEXT NOT NULL,
    prohibited_words TEXT NOT NULL,
    style_guide TEXT NOT NULL,
    PRIMARY KEY (client_id, channel)
);

CREATE TABLE IF NOT EXISTS client_channel_brief (
    client_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    tone TEXT NOT NULL,
    word_length INTEGER NOT NULL,
    PRIMARY KEY (client_id, channel)
);

CREATE TABLE IF NOT EXISTS campaigns (
    thread_id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    channel TEXT,
    campaign_topic TEXT,
    status TEXT NOT NULL,
    chat_history TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_checkpointer_connection() -> sqlite3.Connection:
    """PRD §11.7 -- same file as everything else in content_agent.db (one
    SQLite file to volume-mount in Docker, not two). check_same_thread=False
    because this single long-lived connection is shared across whatever
    thread Gradio's queue happens to service a given request on -- SqliteSaver
    itself serializes access, so this is safe."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
