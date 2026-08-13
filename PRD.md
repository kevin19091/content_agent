# PRD: Agentic Content Generation App (WhatsApp + Push)

## 1. Overview

A standalone LangGraph-based multi-agent system that generates channel-specific
marketing content (WhatsApp, App Push) from a rough campaign topic, applies
brand/compliance checks, and routes to a human for sync approval or edits
before finalizing content.

**Scope (from content pipeline stages 2-6):**
- Ideation & briefing
- Creation / drafting
- Review & compliance
- Approval workflow (HITL)

Out of scope for v1: strategy/planning, localization, scheduling/publishing,
distribution, analytics.

---

## 2. Goals

- Given `{client_id, channel, campaign_topic}`, produce approved,
  channel-compliant content ready to hand off to a delivery system.
- Keep human-in-the-loop as a synchronous checkpoint — no content goes out
  without explicit approval.
- Clean separation of concerns across 3 agents so each is independently
  testable and swappable.

## 3. Non-Goals

- Multi-tenant isolation infra (fresh standalone build, single-tenant logic
  is fine for v1; design should not *preclude* multi-tenancy later)
- Actual message delivery via WhatsApp Business API / push provider SDKs
- Localization / multi-language variants
- Analytics / performance feedback loop

---

## 4. Architecture

### 4.1 Agents

| Agent | Responsibility | Tools |
|---|---|---|
| 1. Ideation Agent | Fetch brand guidelines + brief, choose a content angle | `brand_guidelines_tool(client_id, channel)`, `brief_creation_tool(client_id, channel, campaign_topic)` |
| 2. Content Creation Agent | Generate structured, channel-specific content | `.with_structured_output()` bound to channel schema |
| 3. Compliance Agent | Validate against brand guidelines + channel policy | single `.with_structured_output(ComplianceResult)` call — brand + channel rules passed in as prompt context, no separate rule-based tool functions |

### 4.2 Graph Flow

```
ideation_agent
    → human_review [stage=ideation]
        ├─[approve]→ content_creation_agent
        ├─[edit]→ ideation_agent (inject human_edit_notes)
        └─[reject]→ END (log + notify, no content persisted)

content_creation_agent
    → human_review [stage=creation]
        ├─[approve]→ compliance_agent
        ├─[edit]→ content_creation_agent (inject human_edit_notes)
        └─[reject]→ END (log + notify, no content persisted)

compliance_agent (always hands off — no auto-retry, no severity branch)
    → human_review [stage=compliance]
        ├─[approve]→ END (persist final_content)
        ├─[edit]→ content_creation_agent (inject human_edit_notes)
        └─[reject]→ END (log + notify, no content persisted)
```

- `human_review` is a **single** LangGraph node entered from three points in
  the graph. The conditional edge after it reads `state["stage"]` +
  `state["human_decision"]` to resolve routing — same approve/edit/reject
  pattern at every gate: approve advances to the next agent (or END at the
  last gate), edit loops back to the agent that just ran with
  `human_edit_notes` injected, reject ends the run immediately.
- No auto-retry loop and no revision cap — `compliance_agent` never routes
  on its own; it only raises issues, and every result (any severity) goes
  to a human at the compliance gate.
- HITL uses LangGraph's `interrupt()` / `Command(resume=...)` pattern —
  synchronous, same-thread pause/resume via checkpointer. Validated headless
  against a stub graph (`MemorySaver` + 3-way `stage` routing), including
  nested loops — e.g. edit at the compliance gate sends the draft back to
  `content_creation_agent`, which re-enters `compliance_agent`, which
  re-interrupts at `stage=compliance`.

---

## 5. Data Model / State Schema

```python
from typing import Literal, Optional
from typing_extensions import TypedDict

class ContentRequest(TypedDict):
    client_id: str
    channel: Literal["whatsapp", "push"]
    campaign_topic: str

class AgentState(TypedDict):
    request: ContentRequest
    brand_guidelines: Optional[dict]
    brief: Optional[dict]
    angle: Optional[str]
    draft_content: Optional[dict]
    compliance_result: Optional[dict]
    stage: Optional[Literal["ideation", "creation", "compliance"]]
    human_decision: Optional[Literal["approve", "edit", "reject"]]
    human_edit_notes: Optional[str]
    final_content: Optional[dict]
```

- `stage` is written by whichever agent last ran, right before handing off
  to `human_review` — it's what lets the single shared node know which of
  the three gates it's resolving (see §4.2, §6.4).
- `revision_count` is gone — there's no cap to track since
  `compliance_agent` never auto-retries.

### 5.1 Structured Output Models

```python
from pydantic import BaseModel, Field

class IdeationOutput(BaseModel):
    angle: str = Field(description="chosen content angle/hook")
    tone: str
    key_message: str
    target_audience: str
    constraints: list[str]
    cta: str
    word_length: int

class WhatsAppContent(BaseModel):
    template_name: str
    header: Optional[str] = None
    body: str
    footer: Optional[str] = None
    cta_button_text: Optional[str] = None

class PushContent(BaseModel):
    title: str
    body: str
    cta_button_text: Optional[str] = None

class ComplianceResult(BaseModel):
    passed: bool
    issues: list[str] = []
    severity: Literal["none", "minor", "blocking"]
```

- `Brief` no longer exists as a nested model — its fields sit directly on
  `IdeationOutput` alongside `angle`. `tone` and `word_length` are
  overwritten with DB-looked-up values after the LLM call returns (the LLM
  drafts them but the DB is authoritative — see §6.1); `key_message`,
  `target_audience`, `constraints`, `cta` are always LLM-derived.
- No `Field(max_length=...)` on `WhatsAppContent`/`PushContent` — OpenAI's
  structured-output mode doesn't reliably enforce pydantic validators
  during generation, and a violation raising mid-call was judged higher risk
  than the length limits it was meant to catch. `compliance_agent` is the
  sole enforcer of char/length limits, given the actual numbers (WhatsApp
  body ≤1024, Push title ≤65 / body ≤178) as prompt context.
- `WhatsAppContent.variables` and `PushContent.deep_link`/`icon` are
  dropped for v1 (no template variable substitution, no deep linking);
  `PushContent` gains `cta_button_text` to match WhatsApp's CTA shape.

---

## 6. Node Specifications

### 6.1 `ideation_agent`
- Input: `request`
- Calls `brand_guidelines_tool(client_id, channel)` → guideline dict (tone
  rules, prohibited words, style guide), looked up per (client, channel)
- Calls `brief_creation_tool(client_id, channel, campaign_topic)`:
  - DB lookup keyed on (client_id, channel) → `tone`, `word_length`
  - LLM call from `campaign_topic` → `key_message`, `target_audience`,
    `constraints`, `cta`
  - `tone`/`word_length` from the DB win over anything the LLM proposes
    for those two fields
- Reasons over guidelines + brief fields to select **one** angle (planning
  only, no copy drafting here — keeps Agent 2's structured output clean)
- Output: `IdeationOutput` written to state — `angle` plus the flat brief
  fields (merged into `brief`), and `brand_guidelines`
- → `human_review` [stage=ideation]

### 6.2 `content_creation_agent`
- Input: `angle`, `brief` (incl. `word_length`), `brand_guidelines`,
  `request.channel`, optionally `human_edit_notes` (on an edit loop from
  any gate) or `compliance_result.issues` (on an edit loop from the
  compliance gate specifically)
- Selects schema based on `channel`: `WhatsAppContent` or `PushContent`
- Calls LLM with `.with_structured_output(<schema>)`, prompted with the
  channel's actual length limits (WhatsApp body ≤1024, Push title ≤65 /
  body ≤178) and `brief.word_length` as a target — not enforced by pydantic
  `Field` constraints (see §5.1)
- On revision: prompt includes prior `draft_content` +
  `compliance_result.issues` or `human_edit_notes` as correction context
- Output: `draft_content`
- → `human_review` [stage=creation]

### 6.3 `compliance_agent`
- Input: `draft_content`, `brand_guidelines`, `request.channel`
- Single `.with_structured_output(ComplianceResult)` call — tone match,
  prohibited word scan, channel-specific rules (WhatsApp: template category
  rules, char limits; Push: title/body length limits) are all given to the
  LLM as prompt context rather than implemented as separate rule-based tool
  functions
- Output: `ComplianceResult` → `compliance_result`
- No routing logic here — always falls through to `human_review`
  regardless of `severity`. No auto-retry, no cap.
- → `human_review` [stage=compliance]

### 6.4 `human_review` (interrupt node, shared across 3 stages)
```python
from langgraph.types import interrupt

def human_review(state: AgentState):
    decision = interrupt({
        "stage": state["stage"],
        "angle": state.get("angle"),
        "brief": state.get("brief"),
        "draft_content": state.get("draft_content"),
        "compliance_result": state.get("compliance_result"),
    })
    return {
        "human_decision": decision["action"],
        "human_edit_notes": decision.get("notes"),
    }
```
- Resume via `Command(resume={"action": "approve"})` or
  `{"action": "edit", "notes": "..."}` or `{"action": "reject"}`
- Routing (conditional edge reads `state["stage"]` + `human_decision`):

  | stage | entered from | approve → | edit → | reject → |
  |---|---|---|---|---|
  | `ideation` | `ideation_agent` | `content_creation_agent` | `ideation_agent` | END, no content |
  | `creation` | `content_creation_agent` | `compliance_agent` | `content_creation_agent` | END, no content |
  | `compliance` | `compliance_agent` (always, any severity) | END, `final_content = draft_content` | `content_creation_agent` | END, no content |

- `reject` at any stage: END, log + notify, no `final_content` persisted.

---

## 7. Persistence

- Checkpointer required to support sync interrupt/resume across the
  same thread.
  - v1: `MemorySaver` (in-process, fine for local dev in Claude Code)
  - Later: swap to `SqliteSaver` or `PostgresSaver` for durability across
    restarts
- Guidelines/brief source data: simple SQLite tables for v1, both keyed on
  `(client_id, channel)`:
  - `brand_guidelines`: tone rules, prohibited words, style guide
  - `client_channel_brief`: `tone`, `word_length` (the two fields
    `brief_creation_tool` looks up rather than derives)
  - Static/pre-seeded — no authoring UI for v1
  - (no vector search needed unless RAG over past approved content is
    wanted later)

---

## 8. Tech Stack (v1 defaults — confirm/adjust)

- **Orchestration:** LangGraph (Python)
- **LLM:** OpenAI, via `langchain-openai`, structured output through
  `.with_structured_output()`
- **DB:** SQLite (local dev) → Postgres (later)
- **Checkpointer:** `MemorySaver` → `SqliteSaver`/`PostgresSaver`
- **Interface:** Gradio app, chat-style (WhatsApp-like transcript)
  - Login gate via `gr.Blocks().launch(auth=<callable>)` — `client_name` +
    one shared password for all clients (v1 only, no per-client
    credentials, no auth table)
  - A small structured form (channel + campaign topic) kicks off a run.
    From there the whole `human_review` loop is one continuous
    `gr.Chatbot` transcript: each interrupt's payload renders as a
    bulleted markdown message on the right (`role="user"` in Gradio's
    convention); the human types free text in a single message box,
    rendered on the left (`role="assistant"`) — no approve/edit/reject
    buttons.
  - Each human message triggers one additional LLM call (on top of the
    agents' own calls) that classifies it into approve/edit/reject before
    resuming via `Command(resume=...)`; the raw message is reused verbatim
    as `human_edit_notes` when the result is `edit`.
- **Deployment:** Dockerized (single container running the Gradio app;
  SQLite file on a mounted volume, `OPENAI_API_KEY` passed through env)

---

## 9. Decisions (resolved from v0 open questions)

- **`brief_creation_tool` source:** hybrid, split by field — `tone` and
  `word_length` are always a DB lookup keyed on `(client_id, channel)`;
  `key_message`, `target_audience`, `constraints`, `cta` are always
  LLM-derived from `campaign_topic`. No "full brief already exists" bypass.
- **Brand guidelines:** static, pre-seeded per `(client_id, channel)`,
  read-only for v1 — no authoring UI.
- **Revision cap:** removed entirely, not just reset-on-edit as originally
  recommended. `compliance_agent` never auto-retries — it only raises
  issues, and every result (any severity) goes to a human. Simpler than
  tracking a cap, and there's no `revision_count` field left to reset.
- **LLM provider:** OpenAI, via `langchain-openai`.
- **`IdeationOutput` shape:** flattened — no nested `Brief` model.
- **Compliance implementation:** one LLM call with rules as prompt context,
  not separate rule-based tool functions + LLM — less surface to build and
  debug in a short window.
- **Structured-output field constraints:** dropped (`Field(max_length=...)`
  removed from `WhatsAppContent`/`PushContent`) — a validator raising
  mid-call was judged a bigger risk than the limits it was meant to catch.
  `compliance_agent` is the sole enforcer, via prompt.
- **Auth:** `client_name` + one shared password for all clients, v1 only,
  via Gradio's built-in `launch(auth=...)`.
- **UI:** Gradio, not CLI/FastAPI.
- **Deployment:** Docker.
- **HITL mechanism:** validated headless before building the UI — a stub
  graph (`MemorySaver` + `interrupt()`/`Command(resume=...)`) confirmed the
  single shared `human_review` node correctly resolves approve/edit/reject
  at all three stages, including nested loops (edit at the compliance gate
  → back to `content_creation_agent` → re-runs `compliance_agent` →
  re-interrupts at `stage=compliance`).

---

## 10. Milestones

1. **Scaffold** — state schema (incl. `stage`), stub nodes, graph wiring
   with `MemorySaver`, all 3 `human_review` entry points and the
   approve/edit/reject routing table wired against stub agents. *(De-risked
   headless — see §9; re-verify once real nodes are swapped in.)*
2. **DB + tools** — SQLite schema for `brand_guidelines` and
   `client_channel_brief`, seed data, `brand_guidelines_tool` +
   `brief_creation_tool` implementations
3. **Agent 1** — ideation LLM call against real tools, `IdeationOutput`
   structured output
4. **Agent 2** — structured output for both channels, revision-loop
   prompt handling (human notes + compliance issues as correction context)
5. **Agent 3** — single-call compliance check against brand guidelines +
   channel policy
6. **Gradio UI** — login gate, run kickoff, stage-aware review screen,
   resume wiring
7. **Docker** — containerize, verify SQLite volume + `OPENAI_API_KEY`
   passthrough
8. **End-to-end test** — pytest hitting the graph directly
   (`invoke`/`Command(resume=...)`) across the 3×3 stage/decision matrix
   for both channels, plus one full manual golden-path run per channel
   through the Gradio UI