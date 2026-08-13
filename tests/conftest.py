import os

# ChatOpenAI validates credentials at construction time, and several
# content_agent modules build one at import time. Set a dummy key so the
# whole suite can import/collect without a real OPENAI_API_KEY -- every LLM
# call itself is mocked below or per-test, so this key is never actually used.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")

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

        if "previous angle was rejected" in prompt:
            return _AngleSelection(angle="stub angle (revised)")
        return _AngleSelection(angle="stub angle")


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
