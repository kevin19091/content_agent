import os
import uuid
from typing import Literal, Optional

import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import BaseModel

from content_agent.db import init_db
from content_agent.graph import compile_app
from content_agent.seed import seed

load_dotenv()

_SHARED_PASSWORD = os.environ.get("CONTENT_AGENT_SHARED_PASSWORD", "changeme")
_app = compile_app()


class _DecisionClassification(BaseModel):
    action: Literal["approve", "edit", "reject"]


_decision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_structured_decision_llm = _decision_llm.with_structured_output(_DecisionClassification)

_STAGE_DESCRIPTIONS = {
    "ideation": "the proposed content angle and brief",
    "creation": "a draft of the actual copy",
    "compliance": "a brand/channel compliance check result",
}


def _classify_decision(stage: str, message: str) -> str:
    """Free-text -> approve/edit/reject. The raw message itself is reused
    verbatim as human_edit_notes when the result is 'edit' -- this call
    only decides which of the three actions was meant."""
    prompt = (
        f"A human is reviewing {_STAGE_DESCRIPTIONS.get(stage, 'an agent output')} "
        f"in a content approval workflow and just sent this message:\n\n\"{message}\"\n\n"
        "Classify their intent as exactly one of:\n"
        "- approve: satisfied, move on as-is\n"
        "- edit: wants changes; their message is the feedback to act on\n"
        "- reject: wants to kill this campaign entirely, not just request changes"
    )
    return _structured_decision_llm.invoke(prompt).action


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

    if stage == "ideation":
        fields = {"angle": payload.get("angle"), **(payload.get("brief") or {})}
        return "**Proposed angle & brief**\n" + _bullets(fields)

    if stage == "creation":
        return "**Draft content**\n" + _bullets(payload.get("draft_content") or {})

    if stage == "compliance":
        cr = payload.get("compliance_result") or {}
        lines = [f"- **Passed:** {cr.get('passed')}", f"- **Severity:** {cr.get('severity')}"]
        issues = cr.get("issues") or []
        if issues:
            lines.append("- **Issues:**")
            lines.extend(f"  - {issue}" for issue in issues)
        else:
            lines.append("- **Issues:** none")
        return "**Compliance review**\n" + "\n".join(lines)

    return "**Update**\n" + _bullets(payload)


def _authenticate(username: str, password: str) -> bool:
    """PRD §8, §9 -- client_name + one shared password for all clients, v1
    only. The username becomes client_id for the session; it isn't checked
    against the DB here -- an unknown client surfaces as a clear error when
    a campaign is started instead (see _start_campaign)."""
    return bool(username) and password == _SHARED_PASSWORD


def _append_result(result: dict, thread_id: str, history: list):
    """Agent turns render on the right (role='user' in Gradio's chat
    convention) -- human turns render on the left (role='assistant')."""
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        history = history + [{"role": "user", "content": _format_agent_message(payload)}]
        return history, thread_id, payload["stage"], ""

    if result.get("final_content") is not None:
        msg = "✅ **Approved** -- final content persisted, ready for delivery:\n" + _bullets(result["final_content"])
    else:
        msg = "⛔ **Rejected** -- no content persisted. Logged + client notified."
    history = history + [{"role": "user", "content": msg}]
    return history, thread_id, None, ""


def _start_campaign(channel: str, campaign_topic: str, request: gr.Request):
    if not campaign_topic or not campaign_topic.strip():
        raise gr.Error("Campaign topic is required.")

    client_id = request.username
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    history = [{"role": "assistant", "content": f"Start a **{channel}** campaign: {campaign_topic}"}]

    try:
        result = _app.invoke(
            {"request": {"client_id": client_id, "channel": channel, "campaign_topic": campaign_topic}},
            config=config,
        )
    except ValueError as e:
        raise gr.Error(str(e))

    return _append_result(result, thread_id, history)


def _submit_message(message: str, thread_id: Optional[str], stage: Optional[str], history: list):
    if not thread_id or not stage:
        raise gr.Error("Start a campaign first.")
    if not message or not message.strip():
        raise gr.Error("Type a message first.")

    history = history + [{"role": "assistant", "content": message}]

    action = _classify_decision(stage, message)
    resume = {"action": action}
    if action == "edit":
        resume["notes"] = message

    config = {"configurable": {"thread_id": thread_id}}
    result = _app.invoke(Command(resume=resume), config=config)

    return _append_result(result, thread_id, history)


def _on_load(request: gr.Request):
    return f"Signed in as **{request.username}** (used as `client_id`)"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Content Agent") as demo:
        gr.Markdown("# Content Agent")
        client_id_md = gr.Markdown()

        thread_state = gr.State(None)
        stage_state = gr.State(None)

        with gr.Accordion("New campaign", open=True):
            channel_dd = gr.Dropdown(["whatsapp", "push"], value="whatsapp", label="Channel")
            topic_tb = gr.Textbox(label="Campaign topic", placeholder="e.g. end of season clearance sale")
            start_btn = gr.Button("Start campaign", variant="primary")

        chatbot = gr.Chatbot(type="messages", label="Review", height=500)

        with gr.Row():
            msg_tb = gr.Textbox(
                placeholder='Message the agent... e.g. "approve", "make it punchier", "reject this"',
                scale=8,
                show_label=False,
            )
            send_btn = gr.Button("Send", scale=1)

        demo.load(_on_load, outputs=[client_id_md])

        io = [chatbot, thread_state, stage_state, msg_tb]
        start_btn.click(_start_campaign, inputs=[channel_dd, topic_tb], outputs=io)
        msg_tb.submit(_submit_message, inputs=[msg_tb, thread_state, stage_state, chatbot], outputs=io)
        send_btn.click(_submit_message, inputs=[msg_tb, thread_state, stage_state, chatbot], outputs=io)

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
