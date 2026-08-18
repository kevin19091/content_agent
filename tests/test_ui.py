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


def test_format_agent_message_ideation_shows_three_options_with_recommendation_flagged():
    payload = {
        "stage": "ideation",
        "angle": "big savings",
        "angle_options": ["big savings", "limited time", "exclusive access"],
        "brief": {"tone": "warm", "cta": "shop now"},
    }
    msg = _format_agent_message(payload)
    assert "Proposed angles & brief" in msg
    assert "1. big savings  *(recommended)*" in msg
    assert "2. limited time" in msg
    assert "2. limited time  *(recommended)*" not in msg
    assert "3. exclusive access" in msg
    assert "- **Tone:** warm" in msg


def test_format_agent_message_ideation_falls_back_without_angle_options():
    payload = {"stage": "ideation", "angle": "big savings", "brief": {}}
    msg = _format_agent_message(payload)
    assert "**Angle:** big savings" in msg


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


def test_format_agent_message_node_error_gets_distinct_prefix():
    payload = {"stage": "creation", "draft_content": None, "node_error": "rate limited"}
    msg = _format_agent_message(payload)
    assert "Something went wrong" in msg
    assert "rate limited" in msg
    assert msg.index("Something went wrong") < msg.index("Draft content")


def test_format_agent_message_no_node_error_has_no_warning_prefix():
    payload = {"stage": "creation", "draft_content": {"body": "x"}}
    msg = _format_agent_message(payload)
    assert "Something went wrong" not in msg


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


class _AlwaysFailsLLM:
    """RuntimeError is explicitly non-retryable per langgraph's
    default_retry_on, so this fails on the first attempt -- fast test, no
    real RetryPolicy backoff wait."""

    def invoke(self, prompt):
        raise RuntimeError("simulated content_creation_agent failure")


class _WorkingCreationLLM:
    def invoke(self, prompt):
        from content_agent.schemas import WhatsAppContent

        return WhatsAppContent(template_name="t", body="recovered draft", cta_button_text="Shop now")


def test_recovers_from_a_node_failure_and_can_continue(monkeypatch):
    """End-to-end through the real graph: content_creation_agent fails on
    its first invocation, RetryPolicy exhausts, ui.py's
    _recover_from_failure lands back on human_review's interrupt with
    node_error visible and distinctly styled. classify_decision's safety
    guard then forces a retry of the failed step regardless of what the
    human says (draft_content was never actually produced) -- swapping
    back to a working LLM proves the retry actually succeeds and the run
    continues normally."""
    history, thread_id, stage, _ = _start_campaign("whatsapp", "sale", FakeRequest())
    assert stage == "ideation"

    # break content_creation_agent BEFORE it's ever invoked -- approving
    # here is what first routes into it.
    monkeypatch.setattr(
        "content_agent.nodes.creation._structured_llms",
        {"whatsapp": _AlwaysFailsLLM(), "push": _AlwaysFailsLLM()},
    )
    history, thread_id, stage, _ = _submit_message("approve", thread_id, stage, history)
    assert stage == "creation"  # re-interrupted at the stage it would have produced
    assert "Something went wrong" in history[-1]["content"]
    assert "simulated content_creation_agent failure" in history[-1]["content"]

    monkeypatch.setattr(
        "content_agent.nodes.creation._structured_llms",
        {"whatsapp": _WorkingCreationLLM(), "push": _WorkingCreationLLM()},
    )
    history, thread_id, stage, _ = _submit_message("okay try again", thread_id, stage, history)
    assert stage == "creation"
    assert "Something went wrong" not in history[-1]["content"]
    assert "recovered draft" in history[-1]["content"]
