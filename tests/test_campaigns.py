from content_agent.campaigns import get_campaign, list_campaigns, save_campaign


def test_save_and_get_round_trips_chat_history():
    history = [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "hello"}]
    save_campaign("t1", "acme", "push", "flash sale", "in_progress", history)

    campaign = get_campaign("t1")
    assert campaign["thread_id"] == "t1"
    assert campaign["client_id"] == "acme"
    assert campaign["channel"] == "push"
    assert campaign["campaign_topic"] == "flash sale"
    assert campaign["status"] == "in_progress"
    assert campaign["chat_history"] == history


def test_save_allows_null_channel_and_topic_during_intake():
    save_campaign("t2", "acme", None, None, "in_progress", [])
    campaign = get_campaign("t2")
    assert campaign["channel"] is None
    assert campaign["campaign_topic"] is None


def test_save_upserts_same_thread_id_and_preserves_created_at():
    save_campaign("t3", "acme", None, None, "in_progress", [{"role": "assistant", "content": "hi"}])
    first = get_campaign("t3")

    save_campaign("t3", "acme", "push", "sale", "approved", [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "bye"}])
    second = get_campaign("t3")

    assert second["thread_id"] == "t3"
    assert second["channel"] == "push"
    assert second["status"] == "approved"
    assert len(second["chat_history"]) == 2
    assert second["created_at"] == first["created_at"]  # preserved across the upsert


def test_get_campaign_returns_none_for_unknown_thread_id():
    assert get_campaign("nonexistent-thread") is None


def test_list_campaigns_scoped_to_client_and_most_recent_first():
    save_campaign("a1", "acme", "push", "sale one", "in_progress", [])
    save_campaign("a2", "acme", "whatsapp", "sale two", "approved", [])
    save_campaign("b1", "other-client", "push", "not visible to acme", "in_progress", [])

    campaigns = list_campaigns("acme")
    thread_ids = [c["thread_id"] for c in campaigns]
    assert "a1" in thread_ids
    assert "a2" in thread_ids
    assert "b1" not in thread_ids
    # most recently updated (a2, saved second) first
    assert thread_ids.index("a2") < thread_ids.index("a1")


def test_list_campaigns_empty_for_unknown_client():
    assert list_campaigns("nobody-with-this-id") == []
