from content_agent.nodes.compliance import compliance_agent
from content_agent.schemas import ComplianceResult
from content_agent.state import AgentState


def _base_state(channel="whatsapp", **overrides) -> AgentState:
    state: AgentState = {
        "request": {"client_id": "acme", "channel": channel, "campaign_topic": "end of season sale"},
        "brand_guidelines": {
            "tone_rules": ["friendly"],
            "prohibited_words": ["guarantee", "free money"],
            "style_guide": "be nice",
        },
        "brief": None,
        "angle": None,
        "draft_content": {"template_name": "t", "body": "Shop now and save!"},
        "compliance_result": None,
        "stage": None,
        "human_decision": None,
        "human_edit_notes": None,
        "final_content": None,
    }
    state.update(overrides)
    return state


def test_returns_compliance_result_shape_and_sets_stage():
    result = compliance_agent(_base_state())
    assert result["stage"] == "compliance"
    assert set(result["compliance_result"].keys()) == {"passed", "issues", "severity"}


def test_never_produces_routing_decision():
    """compliance_agent has no conditional logic of its own -- it always
    hands off to human_review regardless of what it finds (PRD §9)."""
    result = compliance_agent(_base_state())
    assert "human_decision" not in result


def test_prompt_includes_draft_guidelines_and_channel_rules(monkeypatch):
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return ComplianceResult(passed=True, issues=[], severity="none")

    monkeypatch.setattr("content_agent.nodes.compliance._structured_llm", _Recording())
    compliance_agent(_base_state(channel="push", draft_content={"title": "guarantee results", "body": "x"}))

    assert "guarantee" in captured["prompt"]
    assert "title must be at most 65 characters" in captured["prompt"]


def test_prohibited_word_scan_is_computed_not_left_for_the_model_to_find(monkeypatch):
    """Same principle as the length-count fix: exact substring presence
    is 100% computable, so it's computed in Python and stated as a fact,
    not left for the model to re-scan the draft text itself."""
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return ComplianceResult(passed=True, issues=[], severity="none")

    monkeypatch.setattr("content_agent.nodes.compliance._structured_llm", _Recording())

    compliance_agent(
        _base_state(
            channel="push",
            draft_content={"title": "We guarantee big savings", "body": "shop now"},
        )
    )
    assert "FOUND ['guarantee']" in captured["prompt"]
    assert "always a blocking issue" in captured["prompt"]


def test_prohibited_word_scan_reports_clean_when_nothing_found(monkeypatch):
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return ComplianceResult(passed=True, issues=[], severity="none")

    monkeypatch.setattr("content_agent.nodes.compliance._structured_llm", _Recording())

    compliance_agent(_base_state(channel="push", draft_content={"title": "Weekend deals", "body": "shop now"}))
    assert "none of the prohibited words are present" in captured["prompt"]
    assert "FOUND" not in captured["prompt"]


def test_prohibited_word_scan_is_case_insensitive(monkeypatch):
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return ComplianceResult(passed=True, issues=[], severity="none")

    monkeypatch.setattr("content_agent.nodes.compliance._structured_llm", _Recording())

    compliance_agent(_base_state(channel="push", draft_content={"title": "GUARANTEE savings", "body": "x"}))
    assert "FOUND ['guarantee']" in captured["prompt"]


def test_whatsapp_and_push_get_different_channel_rules(monkeypatch):
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return ComplianceResult(passed=True, issues=[], severity="none")

    monkeypatch.setattr("content_agent.nodes.compliance._structured_llm", _Recording())

    compliance_agent(_base_state(channel="whatsapp"))
    whatsapp_prompt = captured["prompt"]

    compliance_agent(_base_state(channel="push"))
    push_prompt = captured["prompt"]

    assert "1024 characters" in whatsapp_prompt
    assert "1024 characters" not in push_prompt
    assert "65 characters" in push_prompt
    assert "65 characters" not in whatsapp_prompt


def test_prompt_states_actual_character_counts_not_left_for_the_model_to_count(monkeypatch):
    """The bug this guards against: the model doesn't reliably count
    characters from raw text (verified live -- it flagged a 29-char title
    as "36 characters, exceeds 65" on a draft nowhere near the limit).
    Real counts must be computed in Python and stated in the prompt."""
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return ComplianceResult(passed=True, issues=[], severity="none")

    monkeypatch.setattr("content_agent.nodes.compliance._structured_llm", _Recording())

    draft = {"title": "A" * 12, "body": "B" * 40}
    compliance_agent(_base_state(channel="push", draft_content=draft))

    assert "exactly 12 characters" in captured["prompt"]
    assert "exactly 40 characters" in captured["prompt"]
    assert "within the limit" in captured["prompt"]
    assert "OVER the limit" not in captured["prompt"]


def test_prompt_flags_over_limit_status_for_genuinely_long_content(monkeypatch):
    captured = {}

    class _Recording:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return ComplianceResult(passed=True, issues=[], severity="none")

    monkeypatch.setattr("content_agent.nodes.compliance._structured_llm", _Recording())

    draft = {"title": "A" * 90, "body": "B" * 40}
    compliance_agent(_base_state(channel="push", draft_content=draft))

    assert "exactly 90 characters (OVER the limit)" in captured["prompt"]
    assert "exactly 40 characters (within the limit)" in captured["prompt"]
