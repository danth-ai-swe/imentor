import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.external.fetch_history import fetch_raw_chat_history


def _mock_response(json_payload, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_payload
    resp.raise_for_status.return_value = None
    return resp


def test_returns_messages_from_envelope():
    payload = {
        "success": True,
        "data": {
            "conversationId": 123,
            "messages": [
                {"role": "user",      "content": "hi",     "intent": None,             "createdAt": "2026-04-26T00:00:00Z"},
                {"role": "assistant", "content": "hello",  "intent": "core_knowledge", "createdAt": "2026-04-26T00:00:01Z"},
            ],
        },
    }
    with patch("src.external.fetch_history.httpx.AsyncClient") as client_cls:
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.get.return_value = _mock_response(payload)
        client_cls.return_value = client

        import asyncio
        result = asyncio.run(fetch_raw_chat_history("123"))

    assert result == payload["data"]["messages"]


def test_returns_none_on_unsuccessful_envelope():
    payload = {"success": False, "error": "nope"}
    with patch("src.external.fetch_history.httpx.AsyncClient") as client_cls:
        client = AsyncMock()
        client.__aenter__.return_value = client
        client.get.return_value = _mock_response(payload)
        client_cls.return_value = client

        import asyncio
        result = asyncio.run(fetch_raw_chat_history("123"))

    assert result is None


def test_filter_uses_role_and_flat_content():
    from src.rag.search.entrypoint import _filter_core_knowledge_pairs

    msgs = [
        {"role": "user",      "content": "Q1", "intent": None},
        {"role": "assistant", "content": "A1", "intent": "core_knowledge"},
        {"role": "user",      "content": "Q2", "intent": None},
        {"role": "assistant", "content": "A2", "intent": "off_topic"},
    ]
    out = _filter_core_knowledge_pairs(msgs)
    assert out == [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
