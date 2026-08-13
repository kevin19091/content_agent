import gradio as gr
import pytest

from content_agent.ui import _bullets, _format_agent_message, _start_campaign, _submit_message


class FakeRequest:
    username = "acme"


def test_bullets_formats_dict_skips_none_and_joins_lists():
    result = _bullets({"a": "x", "b": None, "c": ["one", "two"], "d": 5})
    assert "- **A:** x" in result
    assert "**B:**" not in result
    assert "- **C:** one; two" in result
    assert "- **D:** 5" in result


def test_bullets_empty_dict():
    assert _bullets({}) == "- (none)"


def test_format_agent_message_ideation():
    payload = {"stage": "ideation", "angle": "big savings", "brief": {"tone": "warm", "cta": "shop now"}}
    msg = _format_agent_message(payload)
    assert "Proposed angle & brief" in msg
    assert "- **Angle:** big savings" in msg
    assert "- **Tone:** warm" in msg


def test_format_agent_message_creation():
    payload = {"stage": "creation", "draft_content": {"title": "Sale!", "body": "50% off"}}
    msg = _format_agent_message(payload)
    assert "Draft content" in msg
    assert "- **Title:** Sale!" in msg


def test_format_agent_message_compliance_with_issues():
    payload = {
        "stage": "compliance",
        "compliance_result": {"passed": False, "severity": "blocking", "issues": ["too long", "prohibited word"]},
    }
    msg = _format_agent_message(payload)
    assert "Compliance review" in msg
    assert "- **Passed:** False" in msg
    assert "  - too long" in msg
    assert "  - prohibited word" in msg


def test_format_agent_message_compliance_no_issues():
    payload = {"stage": "compliance", "compliance_result": {"passed": True, "severity": "none", "issues": []}}
    msg = _format_agent_message(payload)
    assert "- **Issues:** none" in msg


def test_start_campaign_produces_agent_message_on_right():
    history, thread_id, stage, cleared = _start_campaign("whatsapp", "sale", FakeRequest())
    assert history[0]["role"] == "assistant"  # human's kickoff, left
    assert history[1]["role"] == "user"  # agent output, right
    assert stage == "ideation"
    assert thread_id is not None
    assert cleared == ""


def test_start_campaign_requires_topic():
    with pytest.raises(gr.Error):
        _start_campaign("whatsapp", "  ", FakeRequest())


def test_submit_message_requires_active_thread():
    with pytest.raises(gr.Error):
        _submit_message("approve", None, None, [])


def test_submit_message_requires_nonempty_text():
    history, thread_id, stage, _ = _start_campaign("whatsapp", "sale", FakeRequest())
    with pytest.raises(gr.Error):
        _submit_message("   ", thread_id, stage, history)


def test_free_text_approve_chain_reaches_final_content():
    history, thread_id, stage, _ = _start_campaign("push", "flash sale", FakeRequest())
    assert stage == "ideation"

    history, thread_id, stage, _ = _submit_message("looks great, approve", thread_id, stage, history)
    assert stage == "creation"
    assert history[-2]["role"] == "assistant"  # human message, left
    assert history[-1]["role"] == "user"  # next agent output, right

    history, thread_id, stage, _ = _submit_message("approve", thread_id, stage, history)
    assert stage == "compliance"

    history, thread_id, stage, _ = _submit_message("approve", thread_id, stage, history)
    assert stage is None  # run finished
    assert "Approved" in history[-1]["content"]


def test_free_text_edit_loops_back():
    history, thread_id, stage, _ = _start_campaign("whatsapp", "sale", FakeRequest())
    history, thread_id, stage, _ = _submit_message("approve", thread_id, stage, history)
    assert stage == "creation"

    history, thread_id, stage, _ = _submit_message("please make it shorter and punchier", thread_id, stage, history)
    assert stage == "creation"  # looped back, re-interrupted at same stage
    assert "revised" in history[-1]["content"]


def test_free_text_reject_ends_run_with_no_final_content():
    history, thread_id, stage, _ = _start_campaign("push", "sale", FakeRequest())
    history, thread_id, stage, _ = _submit_message("cancel this, reject it", thread_id, stage, history)
    assert stage is None
    assert "Rejected" in history[-1]["content"]
