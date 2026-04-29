import asyncio
from unittest.mock import AsyncMock, patch

from src.apis.app_model import ChatSourceModel
from src.core.quiz.quiz_chat.intent_classifier import IntentResult
from src.core.quiz.quiz_chat.model import (
    CurrentQuestion,
    OptionItem,
    QuizChatRequest,
)
from src.core.quiz.quiz_chat.pipeline import (
    _aquiz_generate_answer,
    async_quiz_chat_dispatch,
)


def _payload(message: str = "A") -> QuizChatRequest:
    return QuizChatRequest(
        message=message,
        current_question=CurrentQuestion(
            question_type="Definition / Concept",
            difficulty="Beginner",
            question="What is the definition of risk?",
            options=[
                OptionItem(index="A", text="The possibility of gain or profit."),
                OptionItem(index="B", text="The possibility of loss or an undesirable outcome."),
                OptionItem(index="C", text="The certainty of a favorable outcome."),
                OptionItem(index="D", text="The assessment of financial investments."),
            ],
            category="Risk Concepts",
            node_name="Risk",
        ),
    )


def test_hint_intent_returns_hint_only():
    with patch(
        "src.core.quiz.quiz_chat.pipeline.aclassify_intent",
        new=AsyncMock(return_value=IntentResult(intent="hint")),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_chat_client", return_value=object(),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_embedding_client", return_value=object(),
    ):
        result = asyncio.run(async_quiz_chat_dispatch(_payload("hint")))

    assert result.intent == "hint"
    assert result.answer_index is None
    assert result.content is None


def test_finish_intent_returns_finish_only():
    with patch(
        "src.core.quiz.quiz_chat.pipeline.aclassify_intent",
        new=AsyncMock(return_value=IntentResult(intent="finish")),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_chat_client", return_value=object(),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_embedding_client", return_value=object(),
    ):
        result = asyncio.run(async_quiz_chat_dispatch(_payload("finish")))

    assert result.intent == "finish"


def test_answer_intent_with_index():
    with patch(
        "src.core.quiz.quiz_chat.pipeline.aclassify_intent",
        new=AsyncMock(return_value=IntentResult(intent="answer", answer_index="B")),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_chat_client", return_value=object(),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_embedding_client", return_value=object(),
    ):
        result = asyncio.run(async_quiz_chat_dispatch(_payload("loss")))

    assert result.intent == "answer"
    assert result.answer_index == "B"
    assert result.message is None


def test_answer_intent_with_null_index_includes_retry_message():
    with patch(
        "src.core.quiz.quiz_chat.pipeline.aclassify_intent",
        new=AsyncMock(return_value=IntentResult(intent="answer", answer_index=None)),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_chat_client", return_value=object(),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_embedding_client", return_value=object(),
    ):
        result = asyncio.run(async_quiz_chat_dispatch(_payload("hmm not sure")))

    assert result.intent == "answer"
    assert result.answer_index is None
    assert result.message and "match" in result.message.lower()


def test_question_intent_off_topic_route_uses_off_topic_response():
    with patch(
        "src.core.quiz.quiz_chat.pipeline.aclassify_intent",
        new=AsyncMock(return_value=IntentResult(intent="question")),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._avalidate_and_prepare",
        new=AsyncMock(return_value=("English", "what is risk", None)),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._aroute_intent",
        new=AsyncMock(return_value="off_topic"),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.OFF_TOPIC_RESPONSE_MAP",
        {"English": "Sorry, that's off topic."},
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_chat_client", return_value=object(),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_embedding_client", return_value=object(),
    ):
        result = asyncio.run(async_quiz_chat_dispatch(_payload("what is risk?")))

    assert result.intent == "question"
    assert result.content == "Sorry, that's off topic."
    assert result.sources == []


def test_question_intent_validate_error_returns_error_message():
    with patch(
        "src.core.quiz.quiz_chat.pipeline.aclassify_intent",
        new=AsyncMock(return_value=IntentResult(intent="question")),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._avalidate_and_prepare",
        new=AsyncMock(return_value=(None, None, "Input too long")),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_chat_client", return_value=object(),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_embedding_client", return_value=object(),
    ):
        result = asyncio.run(async_quiz_chat_dispatch(_payload("blah")))

    assert result.intent == "question"
    assert result.content == "Input too long"


def test_question_intent_core_route_invokes_quiz_aware_generator():
    captured = {}

    async def fake_quiz_generate(llm, q, lang, chunks, current_question):
        captured["lang"] = lang
        captured["question"] = current_question.question
        return "Risk relates to uncertainty of outcomes."

    fake_source = ChatSourceModel(
        name="loma.pdf", url="http://x/loma.pdf", page_number=1, total_pages=10,
    )

    with patch(
        "src.core.quiz.quiz_chat.pipeline.aclassify_intent",
        new=AsyncMock(return_value=IntentResult(intent="question")),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._avalidate_and_prepare",
        new=AsyncMock(return_value=("English", "what is risk", None)),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._aroute_intent",
        new=AsyncMock(return_value="core_knowledge"),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_qdrant_client", return_value=object(),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._acheck_input_clarity",
        new=AsyncMock(return_value={"clear": True}),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._ahyde_generate",
        new=AsyncMock(return_value="risk hypothetical doc"),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._aembed_text",
        new=AsyncMock(return_value=([0.0], [[0.0]])),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._avector_search",
        new=AsyncMock(return_value=[{"id": "1", "metadata": {}, "text": "chunk", "score": 1.0}]),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.arerank_chunks",
        new=AsyncMock(return_value=[{"id": "1", "metadata": {}, "text": "chunk", "score": 1.0}]),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._afetch_neighbor_chunks",
        new=AsyncMock(return_value=[]),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._merge_chunks",
        return_value=[{"id": "1", "metadata": {}, "text": "chunk", "score": 1.0}],
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._aquiz_generate_answer",
        new=AsyncMock(side_effect=fake_quiz_generate),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.extract_sources",
        return_value=[fake_source],
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_chat_client", return_value=object(),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_embedding_client", return_value=object(),
    ):
        result = asyncio.run(async_quiz_chat_dispatch(_payload("what is risk?")))

    assert result.intent == "question"
    assert result.content == "Risk relates to uncertainty of outcomes."
    assert result.sources == [fake_source]
    assert captured == {"lang": "English", "question": "What is the definition of risk?"}


def test_question_intent_validates_with_no_history():
    captured = {}

    async def fake_validate(message, conversation_id, llm):
        captured["conversation_id"] = conversation_id
        return ("English", "what is risk", None)

    with patch(
        "src.core.quiz.quiz_chat.pipeline.aclassify_intent",
        new=AsyncMock(return_value=IntentResult(intent="question")),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._avalidate_and_prepare",
        new=AsyncMock(side_effect=fake_validate),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline._aroute_intent",
        new=AsyncMock(return_value="off_topic"),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.OFF_TOPIC_RESPONSE_MAP", {"English": "off"},
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_chat_client", return_value=object(),
    ), patch(
        "src.core.quiz.quiz_chat.pipeline.get_openai_embedding_client", return_value=object(),
    ):
        asyncio.run(async_quiz_chat_dispatch(_payload("what is risk?")))

    assert captured == {"conversation_id": None}


def test_aquiz_generate_answer_injects_guard_instruction_into_prompt():
    """Verifies the no-spoiler guarantee: the prompt sent to the LLM must
    contain the QUESTION_GUARD_INSTRUCTION text, the node_name, and every
    option, so the model is told not to reveal the correct answer.
    """
    captured_prompts: list[str] = []

    class FakeLLM:
        async def ainvoke_creative(self, prompt: str) -> str:
            captured_prompts.append(prompt)
            return "Risk is the chance of an undesirable event."

    payload = _payload("what is risk?")
    chunks = [
        {
            "id": "1",
            "metadata": {"file_name": "loma.pdf", "page_number": 1, "total_pages": 31},
            "text": "Risk in insurance refers to the possibility of loss.",
            "score": 1.0,
            "payload": {"file_name": "loma.pdf", "page_number": 1},
        }
    ]

    answer = asyncio.run(_aquiz_generate_answer(
        FakeLLM(),
        standalone_query="what is risk",
        detected_language="English",
        enriched_chunks=chunks,
        current_question=payload.current_question,
    ))

    assert answer == "Risk is the chance of an undesirable event."
    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]

    # Guard instruction body present
    assert "QUIZ CONTEXT" in prompt
    assert "Do NOT state which option" in prompt
    assert "Do NOT echo any single option verbatim" in prompt
    # Question metadata present
    assert payload.current_question.node_name in prompt
    assert payload.current_question.category in prompt
    assert payload.current_question.question in prompt
    # Every option text present so the LLM can recognise what NOT to reveal
    for opt in payload.current_question.options:
        assert opt.text in prompt


def test_aquiz_generate_answer_returns_empty_on_llm_failure():
    """Reviewer I-3: covers the previously-untested exception fallback."""

    class BrokenLLM:
        async def ainvoke_creative(self, prompt: str) -> str:
            raise RuntimeError("LLM down")

    answer = asyncio.run(_aquiz_generate_answer(
        BrokenLLM(),
        standalone_query="what is risk",
        detected_language="English",
        enriched_chunks=[],
        current_question=_payload().current_question,
    ))
    assert answer == ""
