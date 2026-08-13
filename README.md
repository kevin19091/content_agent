# Content Agent

A LangGraph-based multi-agent system that turns a rough campaign topic into
approved, channel-compliant marketing content for WhatsApp and App Push.
Three agents (ideation → creation → compliance) do the work; a human
reviews and approves, edits, or rejects at each stage before anything is
considered final. See [PRD.md](PRD.md) for the full spec and decisions log.

## How it works

- **`ideation_agent`** — looks up brand guidelines + a per-channel brief
  (tone/word length from a DB, key message/audience/CTA from an LLM), then
  picks one content angle.
- **`content_creation_agent`** — drafts structured, channel-specific copy
  (`WhatsAppContent` or `PushContent`) via `.with_structured_output()`.
- **`compliance_agent`** — one LLM call checking brand + channel rules
  (tone, prohibited words, length limits). It never retries on its own —
  every result, any severity, goes to a human.
- **`human_review`** — a single shared node entered after all three agents.
  Approve advances to the next agent (or finalizes, at the last gate); edit
  loops back to the agent that just ran with your notes; reject ends the
  run immediately. No content is ever finalized without explicit approval.

Runs pause/resume synchronously via LangGraph's `interrupt()` /
`Command(resume=...)`, backed by an in-memory checkpointer.

The UI is a single chat transcript, WhatsApp-style: every agent output
renders on the right as bulleted markdown, every human message on the
left. There are no approve/edit/reject buttons — you just type what you
mean ("looks good", "make it punchier", "kill this one") and an LLM call
classifies your intent before resuming the graph.

## Project layout

```
content_agent/
├── state.py              AgentState, ContentRequest
├── routing.py             stage × decision routing table
├── graph.py                graph wiring
├── schemas.py               structured-output models (content + compliance)
├── db.py, seed.py            SQLite: brand guidelines + brief config
├── nodes/                     ideation, creation, compliance, human_review
├── tools/                      brand_guidelines_tool, brief_creation_tool
├── observability.py             Opik tracing, gated on OPIK_API_KEY (no-op otherwise)
└── ui.py                         Gradio chat app (login, campaign form, chat transcript)
app.py                            entry point
tests/                              pytest suite (offline, LLM calls mocked)
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set OPENAI_API_KEY (required), optionally
# CONTENT_AGENT_SHARED_PASSWORD (defaults to "changeme" if unset)

python app.py
```

Open `http://localhost:7860`. Log in with username `acme` (the only
seeded client — used as `client_id`) and your shared password. The DB is
created and seeded automatically on first run.

## Run with Docker

```bash
docker build -t content-agent .
docker run -p 7860:7860 --env-file .env -v content_agent_data:/data content-agent
```

The container writes the SQLite DB to `/data` (the mounted volume), so it
persists across restarts. Don't set `CONTENT_AGENT_DB_PATH` in `.env` — it
defaults correctly for each environment on its own (see the comment in
`.env.example`).

## Tests

```bash
source .venv/bin/activate
pytest
```

All tests run offline — every LLM call is mocked in `tests/conftest.py`,
so no `OPENAI_API_KEY` is needed to run the suite.

## Observability (Opik)

Optional, off by default. Set `OPIK_API_KEY` (and `OPIK_WORKSPACE`) in
`.env` from a free [comet.com](https://www.comet.com) account to trace
every LLM call — both `ideation_agent` calls, `content_creation_agent`'s,
`compliance_agent`'s, and the UI's free-text approve/edit/reject
classifier — grouped by campaign via LangGraph's `thread_id`. Leave
`OPIK_API_KEY` unset and tracing is a true no-op: no network calls, same
behavior the test suite relies on.

## Seeded demo data

Client `acme`, both channels (`whatsapp`, `push`) — see
`content_agent/seed.py`. Brand guidelines and brief config are static and
pre-seeded for v1; there's no authoring UI.
