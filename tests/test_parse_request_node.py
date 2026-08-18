from content_agent.nodes.parse_request import _RequestExtraction, parse_request
from content_agent.state import AgentState


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "client_id": "acme",
        "raw_intake": "push channel, topic: sale",
        "pending_channel": None,
        "pending_campaign_topic": None,
        "intake_cancelled": None,
        "request": None,
        "node_error": None,
        "node_error_source": None,
    }
    state.update(overrides)
    return state


def _fake_llm(channel=None, campaign_topic=None, cancelled=False):
    return type(
        "F",
        (),
        {"invoke": lambda self, p: _RequestExtraction(channel=channel, campaign_topic=campaign_topic, cancelled=cancelled)},
    )()


def test_extracts_both_fields_and_builds_request(monkeypatch):
    monkeypatch.setattr(
        "content_agent.nodes.parse_request._structured_llm", _fake_llm(channel="push", campaign_topic="sale")
    )
    result = parse_request(_base_state())
    assert result["pending_channel"] == "push"
    assert result["pending_campaign_topic"] == "sale"
    assert result["request"] == {"client_id": "acme", "channel": "push", "campaign_topic": "sale"}
    assert result["node_error"] is None
    assert result["node_error_source"] is None


def test_partial_extraction_does_not_build_request(monkeypatch):
    monkeypatch.setattr("content_agent.nodes.parse_request._structured_llm", _fake_llm(campaign_topic="sale"))
    result = parse_request(_base_state(raw_intake="topic: sale"))
    assert result["pending_campaign_topic"] == "sale"
    assert result["pending_channel"] is None
    assert "request" not in result


def test_new_info_merges_with_previously_captured_info(monkeypatch):
    """PRD §11.1 -- accumulates across turns rather than requiring both
    fields restated every time."""
    monkeypatch.setattr("content_agent.nodes.parse_request._structured_llm", _fake_llm(channel="whatsapp"))
    result = parse_request(
        _base_state(raw_intake="whatsapp", pending_channel=None, pending_campaign_topic="end of season sale")
    )
    assert result["pending_channel"] == "whatsapp"
    assert result["pending_campaign_topic"] == "end of season sale"  # carried over, not lost
    assert result["request"] == {
        "client_id": "acme",
        "channel": "whatsapp",
        "campaign_topic": "end of season sale",
    }


def test_extraction_never_overwrites_already_captured_info_with_nothing(monkeypatch):
    """If this turn's message doesn't mention the channel again, the
    already-known channel must survive, not get wiped to None."""
    monkeypatch.setattr("content_agent.nodes.parse_request._structured_llm", _fake_llm(campaign_topic="a new topic"))
    result = parse_request(_base_state(raw_intake="topic: a new topic", pending_channel="push"))
    assert result["pending_channel"] == "push"
    assert result["pending_campaign_topic"] == "a new topic"


def test_cancelled_stops_extraction_and_skips_request(monkeypatch):
    monkeypatch.setattr("content_agent.nodes.parse_request._structured_llm", _fake_llm(cancelled=True))
    result = parse_request(_base_state(raw_intake="never mind, forget it"))
    assert result["intake_cancelled"] is True
    assert "request" not in result
    assert "pending_channel" not in result


def test_client_id_comes_from_state_not_extraction(monkeypatch):
    monkeypatch.setattr(
        "content_agent.nodes.parse_request._structured_llm", _fake_llm(channel="push", campaign_topic="sale")
    )
    result = parse_request(_base_state(client_id="someone-else"))
    assert result["request"]["client_id"] == "someone-else"
