import os
import uuid

import gradio as gr
from dotenv import load_dotenv
from langgraph.types import Command

from content_agent.db import init_db
from content_agent.graph import compile_app
from content_agent.seed import seed

load_dotenv()

_SHARED_PASSWORD = os.environ.get("CONTENT_AGENT_SHARED_PASSWORD", "changeme")
_app = compile_app()


def _authenticate(username: str, password: str) -> bool:
    """PRD §8, §9 -- client_name + one shared password for all clients, v1
    only. The username becomes client_id for the session; it isn't checked
    against the DB here -- an unknown client surfaces as a clear error when
    a campaign is started instead (see _start_campaign)."""
    return bool(username) and password == _SHARED_PASSWORD


def _render(result: dict, thread_id: str):
    """Map a graph result (either an interrupt or a finished run) onto the
    three panels: start / review / done."""
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return (
            gr.update(visible=False),  # start_group
            gr.update(visible=True),  # review_group
            gr.update(visible=False),  # done_group
            f"### Review: `{payload['stage']}` stage",
            payload,
            "",  # clear notes box
            thread_id,
        )

    if result.get("final_content") is not None:
        outcome = "### ✅ Approved\n`final_content` persisted, ready for delivery."
        content = result["final_content"]
    else:
        outcome = "### ⛔ Rejected\nNo content persisted. Logged + client notified."
        content = None

    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        outcome,
        content,
        "",
        thread_id,
    )


def _start_campaign(channel: str, campaign_topic: str, request: gr.Request):
    if not campaign_topic or not campaign_topic.strip():
        raise gr.Error("Campaign topic is required.")

    client_id = request.username
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = _app.invoke(
            {"request": {"client_id": client_id, "channel": channel, "campaign_topic": campaign_topic}},
            config=config,
        )
    except ValueError as e:
        raise gr.Error(str(e))

    return _render(result, thread_id)


def _submit_decision(action: str, notes: str, thread_id: str):
    if action == "edit" and not (notes and notes.strip()):
        raise gr.Error("Notes are required when choosing Edit.")

    config = {"configurable": {"thread_id": thread_id}}
    resume = {"action": action}
    if action == "edit":
        resume["notes"] = notes

    result = _app.invoke(Command(resume=resume), config=config)
    return _render(result, thread_id)


def _reset():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
        None,
        "",
        None,
    )


def _on_load(request: gr.Request):
    return f"Signed in as **{request.username}** (used as `client_id`)"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Content Agent") as demo:
        gr.Markdown("# Content Agent")
        client_id_md = gr.Markdown()

        thread_state = gr.State(None)

        with gr.Group() as start_group:
            gr.Markdown("### New campaign")
            channel_dd = gr.Dropdown(["whatsapp", "push"], value="whatsapp", label="Channel")
            topic_tb = gr.Textbox(label="Campaign topic", placeholder="e.g. end of season clearance sale")
            start_btn = gr.Button("Start campaign", variant="primary")

        with gr.Group(visible=False) as review_group:
            stage_md = gr.Markdown()
            payload_json = gr.JSON(label="Details")
            notes_tb = gr.Textbox(label="Edit notes", placeholder="Required only if choosing Edit")
            with gr.Row():
                approve_btn = gr.Button("Approve", variant="primary")
                edit_btn = gr.Button("Edit")
                reject_btn = gr.Button("Reject", variant="stop")

        with gr.Group(visible=False) as done_group:
            outcome_md = gr.Markdown()
            final_json = gr.JSON(label="Final content")
            new_campaign_btn = gr.Button("Start another campaign")

        outputs = [start_group, review_group, done_group, stage_md, payload_json, notes_tb, thread_state]

        demo.load(_on_load, outputs=[client_id_md])
        start_btn.click(_start_campaign, inputs=[channel_dd, topic_tb], outputs=outputs)
        approve_btn.click(lambda tid: _submit_decision("approve", None, tid), inputs=[thread_state], outputs=outputs)
        edit_btn.click(lambda tid, notes: _submit_decision("edit", notes, tid), inputs=[thread_state, notes_tb], outputs=outputs)
        reject_btn.click(lambda tid: _submit_decision("reject", None, tid), inputs=[thread_state], outputs=outputs)
        new_campaign_btn.click(_reset, outputs=outputs)

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
