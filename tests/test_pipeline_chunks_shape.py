import asyncio
from unittest.mock import AsyncMock, patch

from src.rag.search.pipeline_chunks import async_pipeline_dispatch_chunks


def test_off_topic_returns_static_mode():
    async def run():
        with patch(
            "src.rag.search.pipeline_chunks._avalidate_and_prepare",
            new=AsyncMock(return_value=("English", "hi", None)),
        ), patch(
            "src.rag.search.pipeline_chunks.is_quiz_intent", return_value=False
        ), patch(
            "src.rag.search.pipeline_chunks._aroute_intent",
            new=AsyncMock(return_value="off_topic"),
        ), patch(
            "src.rag.search.pipeline_chunks.get_openai_chat_client", return_value=object()
        ), patch(
            "src.rag.search.pipeline_chunks.get_openai_embedding_client", return_value=object()
        ):
            result = await async_pipeline_dispatch_chunks("hi", None)

        assert result["mode"] == "static"
        assert result["intent"] == "off_topic"
        assert result["chunks"] == []
        assert result["response"]
        assert result["sources"] == []

    asyncio.run(run())
