import os

# ChatOpenAI validates credentials at construction time, and several
# content_agent modules build one at import time. Set a dummy key so the
# whole suite can import/collect without a real OPENAI_API_KEY -- every LLM
# call itself is mocked below or per-test, so this key is never actually used.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")

# Force Opik tracing off regardless of what's in the developer's real .env --
# content_agent.observability calls load_dotenv(), which won't override an
# already-set var (even an empty one), so setting this first wins. Without
# this, a real OPIK_API_KEY in .env would make the suite attempt real network
# calls to Opik on every test run.
os.environ.setdefault("OPIK_API_KEY", "")

import pytest

from content_agent.db import init_db
from content_agent.seed import seed
from content_agent.tools.brief import _DerivedBriefFields


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTENT_AGENT_DB_PATH", str(tmp_path / "test.db"))
    import content_agent.db as db_module

    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    init_db()
    seed()
    yield


class _FakeBriefLLM:
    def invoke(self, prompt):
        return _DerivedBriefFields(
            key_message="stub key message",
            target_audience="stub audience",
            constraints=[],
            cta="stub cta",
        )


class _FakeAngleLLM:
    def invoke(self, prompt):
        from content_agent.nodes.ideation import _AngleSelection

        if "needed rework" in prompt:
            return _AngleSelection(angles=["stub angle (revised)", "stub angle B (revised)", "stub angle C (revised)"])
        return _AngleSelection(angles=["stub angle", "stub angle B", "stub angle C"])


class _FakeCreationLLM:
    def __init__(self, channel):
        self._channel = channel

    def invoke(self, prompt):
        from content_agent.schemas import PushContent, WhatsAppContent

        revising = "Revise the draft" in prompt
        body = "stub draft body (revised)" if revising else "stub draft body"
        if self._channel == "whatsapp":
            return WhatsAppContent(template_name="stub_template", body=body, cta_button_text="Shop now")
        return PushContent(title="Stub title", body=body, cta_button_text="Open")


class _FakeComplianceLLM:
    def invoke(self, prompt):
        from content_agent.schemas import ComplianceResult

        return ComplianceResult(passed=True, issues=[], severity="none")


class _FakeDecisionLLM:
    """Crude keyword-based stand-in for the UI's free-text ->
    approve/edit/reject classifier. The real one uses an LLM; tests just
    need something deterministic. Must isolate the human's quoted message
    from the surrounding prompt -- the instructions themselves contain the
    words "approve"/"edit"/"reject", so keyword-matching the whole prompt
    would always hit "reject" first regardless of what was actually said."""

    def invoke(self, prompt):
        from content_agent.nodes.classify_decision import _DecisionClassification

        try:
            message = prompt.split('message:\n\n"', 1)[1].rsplit('"\n\nClassify', 1)[0]
        except IndexError:
            message = prompt
        lowered = message.lower()
        if "reject" in lowered or "cancel" in lowered or "kill" in lowered:
            return _DecisionClassification(action="reject")
        if "approve" in lowered or "looks good" in lowered or "great" in lowered:
            # PRD §11.5: explicit angle-option pick, keyword-based for tests.
            selected_angle_index = None
            if "second" in lowered:
                selected_angle_index = 1
            elif "third" in lowered:
                selected_angle_index = 2
            return _DecisionClassification(action="approve", selected_angle_index=selected_angle_index)
        # PRD §11.3: explicit target when the message names the angle;
        # otherwise None, letting classify_decision's own default apply --
        # covers both code paths (LLM-specified vs. fallback).
        target_stage = "ideation" if ("angle" in lowered or "rethink" in lowered) else None
        return _DecisionClassification(action="edit", target_stage=target_stage)


@pytest.fixture(autouse=True)
def _fake_llms(monkeypatch):
    """Default deterministic stand-ins for every real LLM call so the graph
    /routing tests stay fast and offline. Individual tests can still
    monkeypatch these further for their own assertions."""
    monkeypatch.setattr("content_agent.tools.brief._structured_llm", _FakeBriefLLM())
    monkeypatch.setattr("content_agent.nodes.ideation._structured_llm", _FakeAngleLLM())
    monkeypatch.setattr(
        "content_agent.nodes.creation._structured_llms",
        {"whatsapp": _FakeCreationLLM("whatsapp"), "push": _FakeCreationLLM("push")},
    )
    monkeypatch.setattr("content_agent.nodes.compliance._structured_llm", _FakeComplianceLLM())
    monkeypatch.setattr("content_agent.nodes.classify_decision._structured_llm", _FakeDecisionLLM())
