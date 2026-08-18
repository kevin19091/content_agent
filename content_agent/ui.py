import os
import uuid
from typing import Optional

import gradio as gr
from dotenv import load_dotenv
from langgraph.types import Command

from content_agent.campaigns import get_campaign, list_campaigns, save_campaign
from content_agent.db import init_db
from content_agent.graph import compile_app
from content_agent.seed import seed

load_dotenv()

_SHARED_PASSWORD = os.environ.get("CONTENT_AGENT_SHARED_PASSWORD", "changeme")
_app = compile_app()

_STAGE_BY_NODE = {
    "ideation_agent": "ideation",
    "content_creation_agent": "creation",
    "compliance_agent": "compliance",
}


def _recover_from_failure(exc: Exception, config: dict) -> dict:
    """PRD §11.4 -- called when a node's RetryPolicy exhausts every
    attempt and the exception escapes invoke(). Finds which node was
    executing (no hardcoding), injects the failure as if that node had
    produced it, and continues -- lands back on an interrupt (human_review
    or, during intake, collect_request) through the graph's own edges,
    same rendering path as any normal turn (see _format_agent_message's
    node_error handling)."""
    failed_node = _app.get_state(config).next[0]
    values = {"node_error": str(exc), "node_error_source": failed_node}
    if failed_node in _STAGE_BY_NODE:
        values["stage"] = _STAGE_BY_NODE[failed_node]
    _app.update_state(config, values, as_node=failed_node)
    return _app.invoke(None, config=config)


def _bullets(d: dict) -> str:
    lines = []
    for k, v in d.items():
        if v is None:
            continue
        label = k.replace("_", " ").title()
        if isinstance(v, list):
            if not v:
                continue
            lines.append(f"- **{label}:** {'; '.join(str(x) for x in v)}")
        else:
            lines.append(f"- **{label}:** {v}")
    return "\n".join(lines) if lines else "- (none)"


def _format_agent_message(payload: dict) -> str:
    stage = payload["stage"]

    node_error = payload.get("node_error")
    prefix = f"⚠️ **Something went wrong** (after 3 attempts)\n{node_error}\n\n---\n\n" if node_error else ""

    if stage == "intake":
        pending_channel = payload.get("pending_channel")
        pending_topic = payload.get("pending_campaign_topic")
        if not pending_channel and not pending_topic:
            body = (
                "Tell me about the campaign you'd like to run -- what's it "
                "about, and which channel (WhatsApp or push)?"
            )
        else:
            known = []
            if pending_channel:
                known.append(f"- **Channel:** {pending_channel}")
            if pending_topic:
                known.append(f"- **Topic:** {pending_topic}")
            missing = "channel (WhatsApp or push)" if not pending_channel else "campaign topic"
            body = "Got it so far:\n" + "\n".join(known) + f"\n\nWhat's the {missing}?"
        return prefix + body

    if stage == "ideation":
        angle_options = payload.get("angle_options") or []
        if angle_options:
            angle_lines = [
                f"{i + 1}. {a}" + ("  *(recommended)*" if i == 0 else "") for i, a in enumerate(angle_options)
            ]
            angles_block = "**Angle options:**\n" + "\n".join(angle_lines)
        else:
            angles_block = "**Angle:** " + str(payload.get("angle"))
        return prefix + "**Proposed angles & brief**\n" + angles_block + "\n\n" + _bullets(payload.get("brief") or {})

    if stage == "creation":
        return prefix + "**Draft content**\n" + _bullets(payload.get("draft_content") or {})

    if stage == "compliance":
        cr = payload.get("compliance_result") or {}
        lines = [f"- **Passed:** {cr.get('passed')}", f"- **Severity:** {cr.get('severity')}"]
        issues = cr.get("issues") or []
        if issues:
            lines.append("- **Issues:**")
            lines.extend(f"  - {issue}" for issue in issues)
        else:
            lines.append("- **Issues:** none")
        return prefix + "**Compliance review**\n" + "\n".join(lines)

    return prefix + "**Update**\n" + _bullets(payload)


def _authenticate(username: str, password: str) -> bool:
    """PRD §8, §9 -- client_name + one shared password for all clients, v1
    only. The username becomes client_id for the session."""
    return bool(username) and password == _SHARED_PASSWORD


def _append_result(result: dict, thread_id: str, history: list):
    """Agent turns render on the right (role='user' in Gradio's chat
    convention) -- human turns render on the left (role='assistant').
    Returns thread_id=None once a run finishes (approved/rejected) so the
    next message starts a fresh campaign instead of trying to resume a
    thread with nothing left to resume."""
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        history = history + [{"role": "user", "content": _format_agent_message(payload)}]
        return history, thread_id, ""

    if result.get("final_content") is not None:
        msg = "✅ **Approved** -- final content persisted, ready for delivery:\n" + _bullets(result["final_content"])
    else:
        msg = "⛔ **Rejected** -- no content persisted. Logged + client notified."
    history = history + [{"role": "user", "content": msg}]
    return history, None, ""


def _plain_text(d: dict) -> str:
    """Copy/paste-friendly rendering of final_content -- no markdown, so it
    can go straight into a delivery system (PRD §11.8, §2)."""
    lines = []
    for k, v in d.items():
        if v is None:
            continue
        label = k.replace("_", " ").title()
        if isinstance(v, list):
            if not v:
                continue
            lines.append(f"{label}: {'; '.join(str(x) for x in v)}")
        else:
            lines.append(f"{label}: {v}")
    return "\n".join(lines)


def _final_content_update(final_content: Optional[dict]):
    if final_content is None:
        return gr.update(value="", visible=False)
    return gr.update(value=_plain_text(final_content), visible=True)


def _persist_campaign(config: dict, history: list, result: dict) -> None:
    """PRD §11.7 -- called after every real turn (not on the bare intake
    greeting from _on_load, to avoid littering the list with abandoned
    "empty" campaigns from a page load that never led to a message)."""
    state = _app.get_state(config).values
    client_id = state.get("client_id")
    if not client_id:
        return

    request_data = state.get("request") or {}
    if "__interrupt__" in result:
        status = "in_progress"
    elif result.get("final_content") is not None:
        status = "approved"
    else:
        status = "rejected"

    save_campaign(
        thread_id=config["configurable"]["thread_id"],
        client_id=client_id,
        channel=request_data.get("channel"),
        campaign_topic=request_data.get("campaign_topic"),
        status=status,
        chat_history=history,
    )


def _campaign_choices(client_id: str):
    choices = [
        (f"{c['channel'] or '?'} · {c['campaign_topic'] or '(no topic yet)'} · {c['status']}", c["thread_id"])
        for c in list_campaigns(client_id)
    ]
    return gr.update(choices=choices, value=None)


def _send_message(message: str, thread_id: Optional[str], history: list, request: gr.Request):
    """PRD §11.1 -- one text box for the whole session. No active thread
    -> this message starts a brand new campaign (client_id from login,
    never from free text); otherwise it resumes whatever's currently
    interrupted, whether that's collect_request (intake) or human_review
    (review loop) -- the graph itself resolves which."""
    if not message or not message.strip():
        raise gr.Error("Type a message first.")

    history = history + [{"role": "assistant", "content": message}]

    if thread_id is None:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        try:
            _app.invoke({"client_id": request.username}, config=config)  # reaches collect_request's interrupt
        except Exception as e:
            result = _recover_from_failure(e, config)
            history, thread_id, cleared = _append_result(result, thread_id, history)
            _persist_campaign(config, history, result)
            return (
                history,
                thread_id,
                cleared,
                _campaign_choices(request.username),
                _final_content_update(result.get("final_content")),
            )
    else:
        config = {"configurable": {"thread_id": thread_id}}

    try:
        result = _app.invoke(Command(resume=message), config=config)
    except Exception as e:
        result = _recover_from_failure(e, config)

    history, thread_id, cleared = _append_result(result, thread_id, history)
    _persist_campaign(config, history, result)
    return (
        history,
        thread_id,
        cleared,
        _campaign_choices(request.username),
        _final_content_update(result.get("final_content")),
    )


def _resume_campaign(thread_id: Optional[str]):
    """PRD §11.7 -- loads a past campaign's exact transcript from the
    campaigns table (not re-derived from graph state). Only carries
    thread_id forward if it's still in_progress -- an approved/rejected
    run has no pending interrupt left to resume, so the next message
    should start a fresh campaign instead of trying to resume a dead one.

    final_content isn't stored in the campaigns table (chat_history already
    has it, rendered) -- for an approved campaign it's read back from the
    graph's own checkpoint, which SqliteSaver keeps around after the run
    ends, so the copy/export box (§11.8) still works after a resume."""
    if not thread_id:
        raise gr.Error("Pick a campaign first.")
    campaign = get_campaign(thread_id)
    if campaign is None:
        raise gr.Error("That campaign could not be found.")
    resumable_thread_id = thread_id if campaign["status"] == "in_progress" else None
    final_content = None
    if campaign["status"] == "approved":
        final_content = _app.get_state({"configurable": {"thread_id": thread_id}}).values.get("final_content")
    return campaign["chat_history"], resumable_thread_id, "", _final_content_update(final_content)


def _on_load(request: gr.Request):
    """Eagerly starts a thread and reaches collect_request's interrupt so
    the greeting shows immediately, before the human types anything."""
    client_id_msg = f"Signed in as **{request.username}** (used as `client_id`)"
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = _app.invoke({"client_id": request.username}, config=config)
    except Exception as e:
        result = _recover_from_failure(e, config)
    history, thread_id, _ = _append_result(result, thread_id, [])
    return client_id_msg, history, thread_id, _campaign_choices(request.username), _final_content_update(None)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Content Agent") as demo:
        gr.Markdown("# Content Agent")
        client_id_md = gr.Markdown()

        thread_state = gr.State(None)

        campaign_dd = gr.Dropdown(label="Past campaigns", choices=[], value=None)

        chatbot = gr.Chatbot(label="Review", height=500)

        final_content_tb = gr.Textbox(
            label="Final content -- ready to copy",
            buttons=["copy"],
            interactive=False,
            visible=False,
        )

        with gr.Row():
            msg_tb = gr.Textbox(
                placeholder='Message the agent... e.g. "a whatsapp sale campaign", "approve", "make it punchier"',
                scale=8,
                show_label=False,
            )
            send_btn = gr.Button("Send", scale=1)

        demo.load(_on_load, outputs=[client_id_md, chatbot, thread_state, campaign_dd, final_content_tb])

        io = [chatbot, thread_state, msg_tb, campaign_dd, final_content_tb]
        msg_tb.submit(_send_message, inputs=[msg_tb, thread_state, chatbot], outputs=io)
        send_btn.click(_send_message, inputs=[msg_tb, thread_state, chatbot], outputs=io)

        campaign_dd.change(
            _resume_campaign, inputs=[campaign_dd], outputs=[chatbot, thread_state, msg_tb, final_content_tb]
        )

    return demo


def main():
    init_db()
    seed()
    demo = build_app()
    demo.launch(
        auth=_authenticate,
        server_name=os.environ.get("CONTENT_AGENT_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("CONTENT_AGENT_PORT", "7860")),
    )


if __name__ == "__main__":
    main()
