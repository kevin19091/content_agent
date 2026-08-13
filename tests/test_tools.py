import pytest

from content_agent.tools.brand_guidelines import brand_guidelines_tool
from content_agent.tools.brief import _DerivedBriefFields, brief_creation_tool


def test_brand_guidelines_tool_returns_seeded_data():
    result = brand_guidelines_tool("acme", "whatsapp")
    assert result["tone_rules"] == ["friendly", "concise", "no exclamation spam"]
    assert "guarantee" in result["prohibited_words"]
    assert "CAPS" in result["style_guide"]


def test_brand_guidelines_tool_differs_by_channel():
    whatsapp = brand_guidelines_tool("acme", "whatsapp")
    push = brand_guidelines_tool("acme", "push")
    assert whatsapp["style_guide"] != push["style_guide"]


def test_brand_guidelines_tool_unknown_client_raises():
    with pytest.raises(ValueError):
        brand_guidelines_tool("nonexistent", "whatsapp")


class _FakeStructuredLLM:
    def __init__(self, result=None, raise_if_called=False):
        self._result = result
        self._raise_if_called = raise_if_called

    def invoke(self, prompt):
        if self._raise_if_called:
            raise AssertionError("LLM should not be called")
        return self._result


def test_brief_creation_tool_db_fields_win_over_llm(monkeypatch):
    """tone/word_length must come from the DB regardless of what the LLM call
    returns -- it's never even asked to propose them (PRD §9)."""
    fake_derived = _DerivedBriefFields(
        key_message="Save big this week",
        target_audience="existing customers",
        constraints=["no discount codes"],
        cta="Shop the sale",
    )
    monkeypatch.setattr(
        "content_agent.tools.brief._structured_llm",
        _FakeStructuredLLM(result=fake_derived),
    )

    result = brief_creation_tool("acme", "whatsapp", "end of season sale")

    assert result["tone"] == "warm and conversational"
    assert result["word_length"] == 60
    assert result["key_message"] == "Save big this week"
    assert result["target_audience"] == "existing customers"
    assert result["constraints"] == ["no discount codes"]
    assert result["cta"] == "Shop the sale"


def test_brief_creation_tool_unknown_client_raises(monkeypatch):
    monkeypatch.setattr(
        "content_agent.tools.brief._structured_llm",
        _FakeStructuredLLM(raise_if_called=True),
    )
    with pytest.raises(ValueError):
        brief_creation_tool("nonexistent", "whatsapp", "some topic")
