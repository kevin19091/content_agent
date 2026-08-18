import gradio as gr
import pytest

from content_agent.campaigns import get_campaign
from content_agent.ui import _bullets, _format_agent_message, _resume_campaign, _send_message


class FakeRequest:
    username = "acme"


def _start(message, history=None):
    """PRD §11.1 -- thread_id=None means _send_message starts a fresh
    thread AND immediately resumes it with `message`, since collect_request
    always interrupts first regardless. One-shot intake message ("push
    channel, topic: ...") resolves in a single turn per conftest.py's fake
    extraction LLM, landing on the first ideation interrupt."""
    history, thread_id, cleared, _dd = _send_message(message, None, history or [], FakeRequest())
    return history, thread_id, cleared


def _send(message, thread_id, history):
    history, thread_id, cleared, _dd = _send_message(message, thread_id, history, FakeRequest())
    return history, thread_id, cleared


def test_bullets_formats_dict_skips_none_and_joins_lists():
    result = _bullets({"a": "x", "b": None, "c": ["one", "two"], "d": 5})
    assert "- **A:** x" in result
    assert "**B:**" not in result
    assert "- **C:** one; two" in result
    assert "- **D:** 5" in result


def test_bullets_empty_dict():
    assert _bullets({}) == "- (none)"


def test_format_agent_message_intake_prompts_for_missing_info_first_time():
    payload = {"stage": "intake", "pending_channel": None, "pending_campaign_topic": None}
    msg = _format_agent_message(payload)
    assert "Tell me about the campaign" in msg


def test_format_agent_message_intake_shows_what_is_already_known():
    payload = {"stage": "intake", "pending_channel": None, "pending_campaign_topic": "end of season sale"}
    msg = _format_agent_message(payload)
    assert "- **Topic:** end of season sale" in msg
    assert "channel" in msg.lower()


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


def test_send_message_requires_nonempty_text():
    with pytest.raises(gr.Error):
        _send_message("   ", None, [], FakeRequest())


def test_first_message_starts_a_campaign_and_reaches_ideation():
    history, thread_id, cleared = _start("push channel, topic: flash sale")
    assert history[0]["role"] == "assistant"  # human's kickoff, left
    assert history[1]["role"] == "user"  # agent output, right
    assert thread_id is not None
    assert cleared == ""
    assert "Proposed angles" in history[-1]["content"]


def test_intake_accumulates_across_turns_before_reaching_ideation():
    history, thread_id, _ = _start("topic: end of season sale")
    assert "channel" in history[-1]["content"].lower()

    history, thread_id, _ = _send("whatsapp", thread_id, history)
    assert "Proposed angles" in history[-1]["content"]


def test_free_text_approve_chain_reaches_final_content():
    history, thread_id, _ = _start("push channel, topic: flash sale")

    history, thread_id, _ = _send("looks great, approve", thread_id, history)
    assert history[-2]["role"] == "assistant"  # human message, left
    assert history[-1]["role"] == "user"  # next agent output, right
    assert "Draft content" in history[-1]["content"]

    history, thread_id, _ = _send("approve", thread_id, history)
    assert "Compliance review" in history[-1]["content"]

    history, thread_id, _ = _send("approve", thread_id, history)
    assert thread_id is None  # run finished -- next message starts fresh
    assert "Approved" in history[-1]["content"]


def test_free_text_edit_loops_back():
    history, thread_id, _ = _start("whatsapp channel, topic: sale")
    history, thread_id, _ = _send("approve", thread_id, history)
    assert "Draft content" in history[-1]["content"]

    history, thread_id, _ = _send("please make it shorter and punchier", thread_id, history)
    assert "revised" in history[-1]["content"]  # looped back, re-interrupted at same stage


def test_free_text_reject_ends_run_with_no_final_content():
    history, thread_id, _ = _start("push channel, topic: sale")
    history, thread_id, _ = _send("cancel this, reject it", thread_id, history)
    assert thread_id is None
    assert "Rejected" in history[-1]["content"]


def test_intake_cancel_ends_run_with_no_final_content():
    history, thread_id, _ = _start("actually never mind, forget it")
    assert thread_id is None
    assert "Rejected" in history[-1]["content"]


def test_a_finished_run_lets_the_next_message_start_a_new_campaign():
    history, thread_id, _ = _start("push channel, topic: sale")
    history, thread_id, _ = _send("approve", thread_id, history)
    history, thread_id, _ = _send("approve", thread_id, history)
    history, thread_id, _ = _send("approve", thread_id, history)
    assert thread_id is None

    # next message starts a brand new campaign from scratch
    history, thread_id, _ = _send("whatsapp channel, topic: another sale", thread_id, history)
    assert thread_id is not None
    assert "Proposed angles" in history[-1]["content"]


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
    history, thread_id, _ = _start("whatsapp channel, topic: sale")

    # break content_creation_agent BEFORE it's ever invoked -- approving
    # here is what first routes into it.
    monkeypatch.setattr(
        "content_agent.nodes.creation._structured_llms",
        {"whatsapp": _AlwaysFailsLLM(), "push": _AlwaysFailsLLM()},
    )
    history, thread_id, _ = _send("approve", thread_id, history)
    assert "Something went wrong" in history[-1]["content"]
    assert "simulated content_creation_agent failure" in history[-1]["content"]

    monkeypatch.setattr(
        "content_agent.nodes.creation._structured_llms",
        {"whatsapp": _WorkingCreationLLM(), "push": _WorkingCreationLLM()},
    )
    history, thread_id, _ = _send("okay try again", thread_id, history)
    assert "Something went wrong" not in history[-1]["content"]
    assert "recovered draft" in history[-1]["content"]


# --- conversation memory (PRD §11.7) --------------------------------------


def test_campaign_is_persisted_after_first_message():
    history, thread_id, _ = _start("push channel, topic: flash sale")
    campaign = get_campaign(thread_id)
    assert campaign is not None
    assert campaign["client_id"] == "acme"
    assert campaign["channel"] == "push"
    assert campaign["campaign_topic"] == "flash sale"
    assert campaign["status"] == "in_progress"
    assert campaign["chat_history"] == history


def test_campaign_status_updates_to_approved_on_completion():
    history, thread_id, _ = _start("push channel, topic: sale")
    real_thread_id = thread_id  # captured before the final turn resets it to None
    history, thread_id, _ = _send("approve", thread_id, history)
    history, thread_id, _ = _send("approve", thread_id, history)
    history, thread_id, _ = _send("approve", thread_id, history)
    assert thread_id is None

    campaign = get_campaign(real_thread_id)
    assert campaign["status"] == "approved"
    assert campaign["chat_history"] == history


def test_campaign_dropdown_choices_refresh_after_sending(monkeypatch):
    _, thread_id, _, dd_update = _send_message("push channel, topic: first one", None, [], FakeRequest())
    choices = dd_update["choices"] if isinstance(dd_update, dict) else dd_update.choices
    values = [v for _label, v in choices]
    assert thread_id in values


def test_resume_campaign_loads_stored_history_and_keeps_thread_id_if_in_progress():
    history, thread_id, _ = _start("push channel, topic: flash sale")
    loaded_history, resumed_thread_id, cleared = _resume_campaign(thread_id)
    assert loaded_history == history
    assert resumed_thread_id == thread_id
    assert cleared == ""


def test_resume_campaign_clears_thread_id_if_already_finished():
    history, thread_id, _ = _start("push channel, topic: sale")
    real_thread_id = thread_id  # captured before the final turn resets it to None
    history, thread_id, _ = _send("approve", thread_id, history)
    history, thread_id, _ = _send("approve", thread_id, history)
    history, thread_id, _ = _send("approve", thread_id, history)

    loaded_history, resumed_thread_id, _ = _resume_campaign(real_thread_id)
    assert loaded_history == history
    assert resumed_thread_id is None  # nothing left to resume -- next message starts fresh


def test_resume_campaign_requires_a_selection():
    with pytest.raises(gr.Error):
        _resume_campaign(None)


def test_resume_campaign_errors_on_unknown_thread_id():
    with pytest.raises(gr.Error):
        _resume_campaign("not-a-real-thread-id")
