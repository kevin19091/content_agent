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
- **Observability:** Opik (Opik Cloud, comet.com), gated on `OPIK_API_KEY`
  being set — a true no-op otherwise, so local dev/tests never require an
  Opik account or make network calls. `track_langgraph` wraps the compiled
  graph, capturing every agent LLM call automatically, grouped by
  campaign via LangGraph's own `thread_id`. The UI's free-text
  approve/edit/reject classifier runs outside the graph's `invoke()`, so
  it's traced separately via `@track`.

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

---

## 11. Version 2 Scope (not yet built)

Everything below is scoped and, where noted, verified by prototype — but not
yet implemented against the real app. Captured here before implementation
so none of it gets lost.

### 11.1 Free-text campaign intake (replaces the channel/topic form)

The channel dropdown and campaign-topic textbox are gone. A single text box
starts a campaign — the same box used for the whole review loop afterward.

- New entry pair ahead of `ideation_agent`: `collect_request` (interrupt
  only, captures raw text) → `parse_request` (LLM extraction of `channel` +
  `campaign_topic`, `RetryPolicy`-eligible).
- `parse_request`'s conditional edge: both fields resolved → build
  `request`, proceed to `ideation_agent`. Still missing something → loop
  back to `collect_request` and ask again. **No cap on this loop** — it's
  not a failure, it's a normal conversation, distinct from `RetryPolicy`
  (which is reserved for genuine LLM/API errors on the same node).
- Answers **accumulate** across turns rather than requiring both fields
  restated each time (`pending_channel`, `pending_campaign_topic` merged on
  every pass) — e.g. topic given in turn 1, channel in turn 2, nothing
  re-asked.
- `client_id` still comes from the login username, never extracted from
  free text.
- **Escape hatch:** `parse_request` also recognizes "never mind" / "cancel"
  during intake and ends the loop gracefully — an uncapped ask-again loop
  needs a way out, or it reads as a trap rather than a conversation.

### 11.2 `human_review` / `classify_decision` split

`human_review` becomes interrupt-only — it just pauses and captures the raw
message (`human_message`). All three occurrences (`stage=ideation`,
`creation`, `compliance`) keep behaving identically to v1's shared node.

The free-text → approve/edit/reject classification (previously done in
`ui.py`, outside the graph) moves into a new node, `classify_decision`,
downstream of `human_review`.

**Why it can't just be appended to `human_review`'s own function body**
(this was the original plan, revised after testing it directly): retrying
a node that contains an `interrupt()` call doesn't re-run the code *after*
the interrupt resolves — on failure it collapses back to re-presenting the
same interrupt, with only one real attempt made regardless of
`max_attempts`. Confirmed by prototype. `classify_decision` has to be its
own node for `RetryPolicy` to behave correctly on it.

Side benefit: classification is now inside `_app.invoke()`, so it's
automatically covered by Opik's `track_langgraph` tracing — no separate
`@track` needed, unlike the v1 UI-layer classifier.

### 11.3 Human-led routing (not just one-step-back)

v1's `edit` always routes to whichever agent produced the thing under
review. `classify_decision`'s output schema grows from `{action}` to
`{action, target_stage}` — still a small closed enum (3 stages), still
resolved deterministically by the routing function, not an open-ended
orchestration decision (see §11.9 on why this still doesn't need a
supervisor agent). Lets a human at the compliance gate say "go back and
change the angle" and have it actually reach `ideation_agent`, instead of
being forced through `content_creation_agent` regardless of what was
asked.

### 11.4 Retry + failure recovery

`RetryPolicy(max_attempts=3, retry_on=<custom predicate>)` attached to
every node that calls an LLM: `parse_request`, `ideation_agent`,
`content_creation_agent`, `compliance_agent`, `classify_decision`. Not
`human_review` — it never calls an LLM, and per §11.2 it can't safely
carry a retry policy anyway.

**Custom `retry_on` predicate, not LangGraph's default.** Checked
LangGraph's default predicate against OpenAI's actual exception hierarchy:
`openai.RateLimitError`/`APITimeoutError`/`APIConnectionError` don't
subclass `ConnectionError` or `httpx.HTTPStatusError`, so they only get
retried by falling through to the default's catch-all — which also means
`openai.AuthenticationError`/`BadRequestError` fall through the same way
and get retried too, wasting attempts on something that will never
succeed. Custom predicate retries the genuinely transient OpenAI
exceptions explicitly and explicitly excludes the rest of
`OpenAIError`.

**Recovery once retries are exhausted (proven by prototype, not just
designed):**
1. The exhausted-retry exception propagates out of `_app.invoke(...)`.
2. `ui.py` catches it, calls `app.get_state(config).next[0]` to find out
   *which* node was executing — no hardcoding.
3. `app.update_state(config, {"node_error": str(exc)}, as_node=failed_node)`
   injects the failure as if that node had produced it.
4. `app.invoke(None, config)` continues the run, landing back on
   `human_review`'s interrupt with the error visible — same rendering path
   as any normal review turn, no separate error UI.

Verified this identically for both failure locations: an upstream agent
node, and `classify_decision` itself (mid-resume) — `classify_decision`'s
own conditional edge checks `node_error` first, routing back to
`human_review` to re-prompt, before falling through to normal
stage/decision routing.

**Error messages need distinct styling in the UI.** Because a recovered
failure rides the exact same `human_review` interrupt as a normal review
turn, nothing currently distinguishes "the agent is asking you to review
this" from "this step technically failed 3 times." A human could
reasonably reply "approve" to an error. Anything carrying `node_error`
needs a visibly different treatment (icon, border color) in chat, even
though it's mechanically the same interrupt.

### 11.5 Three angles instead of one

`ideation_agent`'s structured output changes from `angle: str` to
`angles: list[str]` (exactly 3, prompted strongest-first). Minimal blast
radius by design:

- `AgentState.angle` keeps its v1 meaning exactly — the one angle
  `content_creation_agent` actually drafts from. Nothing downstream of
  ideation changes.
- New `angle_options: list[str]` holds all 3, for display only.
  `ideation_agent` sets `angle = angle_options[0]` by default.
- `classify_decision` gains an optional `selected_angle_index: int | None`
  (meaningful only at `stage=ideation` with `action=approve`) — lets a
  human say "let's go with the second one" without that becoming a 4th
  action type or triggering a fresh `ideation_agent` LLM call just to
  re-surface an option it already generated.

### 11.6 Optional tool calling (`ideation_agent`)

`ideation_agent` currently calls `brand_guidelines_tool` and
`brief_creation_tool` unconditionally, in fixed order — a workflow step,
not agentic behavior (see §11.9). Converts to `.bind_tools()` with a small
internal loop (or `create_react_agent`) so the model decides whether it
needs either tool at all — e.g. on a re-run after an edit, it might
reasonably decide the guidelines it already has are still valid and skip
re-fetching. Scoped to this one node; no other node's tool usage changes.

### 11.7 Conversation memory

- **Checkpointer:** `MemorySaver` → `SqliteSaver`. Prerequisite for
  everything else here — v1 loses every campaign on restart.
- **Not MongoDB — relational, same as the rest of the stack.** Campaign
  records have zero shape variance (every row has the same fields); the
  only query needed is an indexed lookup + sort ("campaigns for client X,
  most recent first"); LangGraph's checkpointer support is natively
  relational (`SqliteSaver`/`PostgresSaver`, no first-class Mongo option);
  and it keeps one storage paradigm instead of fragmenting into two for a
  feature that's just an index table sitting next to state that's already
  relational. Consistent with §8's already-stated `SQLite → Postgres` path.
- **New sidecar table**, not derived from LangGraph's checkpoint blobs
  (those are for resuming one `thread_id`, not querying across threads):
  ```sql
  CREATE TABLE campaigns (
      thread_id TEXT PRIMARY KEY,
      client_id TEXT NOT NULL,
      channel TEXT NOT NULL,
      campaign_topic TEXT NOT NULL,
      status TEXT NOT NULL,        -- in_progress | approved | rejected
      chat_history TEXT NOT NULL,  -- JSON: exact chat transcript, for replay
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
  );
  ```
  Written on every turn alongside the graph invoke. `chat_history` stores
  the formatted transcript verbatim (not re-derived from state on load) so
  a resumed campaign displays exactly what the human saw live.
- **UI:** a campaign list, scoped to the logged-in `client_id`, to reopen
  a past or in-progress conversation.

### 11.8 UI polish

- **Confirm before acting on ambiguous/destructive intent** — echo the
  classified action back before an irreversible one (reject) executes,
  rather than silently acting on a possible misread.
- **Stage visibility** — a breadcrumb (Ideation → Creation → Compliance →
  Done) so position in the pipeline isn't inferred from scrollback.
- **Channel preview rendering** — render `draft_content` as an actual
  WhatsApp bubble / push notification mockup instead of bullets, making
  length-limit issues visible rather than stated.
- **Typing indicator** while an LLM call is in flight.
- **Copy/export the final content** — `final_content` (already exists,
  unchanged since v1) currently just sits in the chat as a bulleted
  message with no way to extract it, despite that being the app's whole
  stated purpose (§2: "ready to hand off to a delivery system").
- **Ambient reply hints** — a caption under the input box, keyed by
  `stage`, suggesting what kind of reply is expected. With zero buttons
  anywhere in the UI, there's currently no affordance at all for what to
  type.

### 11.9 Confirmed: no supervisor agent needed

Revisited after §11.1–§11.3 added dynamic intake and human-led routing —
still no. The test stays the same: a supervisor earns its place when the
*set* of possible next steps is open-ended and needs judgment to narrow
down. Nothing here changes that — `classify_decision`'s target is still a
closed 3-stage enum resolved by a deterministic routing function, and
`parse_request`'s tool-optionality (§11.6) is a local loop inside one
node, not orchestration between nodes. Still 3 fixed agents, same graph
topology, human steering among an enumerable set of targets. Would
reconsider only if a future stage's *applicability* itself needed
reasoning (e.g. "only invoke a localization agent for non-English
markets") — not the case here.

---

## 12. Version 2 Issue Breakdown

§11 broken into independently-gradable vertical slices — each one a thin
but complete path through schema, graph, and UI, demoable on its own.
Dependency order below; everything is AFK (no open design question left
requiring a mid-build decision — all resolved above).

1. **Free-text campaign intake** (§11.1, incl. the escape hatch) —
   `collect_request`/`parse_request`, accumulation across turns, uncapped
   loop-back, graceful cancel. *Blocked by: none.*
   Demo: type a message, get asked follow-ups until channel + topic are
   both known; or say "never mind" and it ends cleanly.

2. **`human_review`/`classify_decision` split + retry + failure recovery**
   (§11.2, §11.4, incl. distinct error styling) — the node split,
   `RetryPolicy` with the OpenAI-aware predicate on every LLM node,
   `node_error` recovery via `update_state(as_node=...)`, visibly distinct
   error messages in chat. *Blocked by: none — this is the foundational
   slice; #3 and #4 both extend `classify_decision`'s schema.*
   Demo: force a node to fail, watch it retry automatically, see it
   surface as a visually distinct error message, reply and watch it
   recover.

3. **Human-led routing** (§11.3) — `target_stage` on `classify_decision`,
   routing to any stage instead of one-step-back. *Blocked by: #2.*
   Demo: from the compliance review, say "go back and change the angle,"
   land at `ideation_agent`.

4. **Three angles instead of one** (§11.5) — `angles`/`angle_options` on
   `ideation_agent`, `selected_angle_index` on `classify_decision`.
   *Blocked by: #2.*
   Demo: see 3 numbered angles at the ideation checkpoint, say "the second
   one," watch `content_creation_agent` draft from that one, not the
   recommendation.

5. **Conversation memory** (§11.7) — `SqliteSaver`, the `campaigns`
   sidecar table, client-scoped campaign list + resume. *Blocked by:
   none — technically independent, cleanest done once the graph shape is
   stable.*
   Demo: start a campaign, close the browser, reopen, find it in a list,
   resume exactly where it left off.

6. **Optional tool calling for `ideation_agent`** (§11.6) — `bind_tools` +
   internal loop, model decides whether to call either tool. *Blocked by:
   none.*
   Demo: trace shows the model skipping a re-fetch of guidelines it
   already has on a re-run.

7. **Copy/export final content** (§11.8) — works against the `final_content`
   field that already exists today, unchanged. *Blocked by: none.*
   Demo: approve a campaign, copy the result, paste it somewhere real.

8. **Ambient reply hints** (§11.8) — per-stage caption under the input box.
   *Blocked by: none.*
   Demo: the hint text changes as the campaign moves through stages.

9. **Stage visibility + confirm-before-destructive** (§11.8) — breadcrumb,
   echo the classified action back before reject executes. *Blocked by:
   none.*
   Demo: see current pipeline position at a glance; attempt a reject, see
   it confirmed before it actually runs.

10. **Channel preview rendering** (§11.8) — `draft_content` as an actual
    WhatsApp bubble / push notification mockup. *Blocked by: none.*
    Demo: a draft renders as a realistic message mockup, not bullets.

11. **Typing indicator** (§11.8) — a visible "thinking" state while an LLM
    call is in flight. *Blocked by: none.*
    Demo: a visible indicator appears during the wait between sending a
    message and getting a response.