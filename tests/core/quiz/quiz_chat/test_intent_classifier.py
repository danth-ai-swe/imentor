import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from src.core.quiz.quiz_chat.intent_classifier import (
    IntentResult,
    aclassify_intent,
    classify_intent_rules,
)
from src.core.quiz.quiz_chat.model import CurrentQuestion, OptionItem


@pytest.mark.parametrize("text,expected", [
    ("A", "A"), ("a", "A"), ("B", "B"), ("b", "B"),
    ("C", "C"), ("c", "C"), ("D", "D"), ("d", "D"),
    (" A ", "A"), ("A.", "A"), ("a)", "A"),
])
def test_letter_inputs_resolve_to_answer(text, expected):
    result = classify_intent_rules(text)
    assert result == IntentResult(intent="answer", answer_index=expected)


@pytest.mark.parametrize("text,expected", [
    ("1", "A"), ("2", "B"), ("3", "C"), ("4", "D"),
    (" 1 ", "A"), ("4.", "D"),
])
def test_digit_inputs_resolve_to_answer(text, expected):
    result = classify_intent_rules(text)
    assert result == IntentResult(intent="answer", answer_index=expected)


@pytest.mark.parametrize("text", [
    "finish", "Finish", "FINISH", "end quiz", "end session",
    "submit and finish", "done", "quit", "exit", "stop",
    "kết thúc", "Kết Thúc", "ket thuc", "thoát", "thoat",
    "nộp bài", "nop bai",
])
def test_finish_keywords_resolve(text):
    result = classify_intent_rules(text)
    assert result == IntentResult(intent="finish")


@pytest.mark.parametrize("text", [
    "hint", "Hint", "HINT", "hin",
    "give me a hint", "help",
    "gợi ý", "Gợi Ý", "goi y",
])
def test_hint_keywords_resolve(text):
    result = classify_intent_rules(text)
    assert result == IntentResult(intent="hint")


def test_typo_close_to_finish_via_fuzzy():
    assert classify_intent_rules("finsh") == IntentResult(intent="finish")


def test_hint_with_trailing_punct_resolves():
    assert classify_intent_rules("hint!").intent == "hint"


def test_unrelated_text_returns_none():
    assert classify_intent_rules("what is risk?") is None
    assert classify_intent_rules("the possibility of loss") is None


def test_empty_input_returns_none():
    assert classify_intent_rules("") is None
    assert classify_intent_rules("   ") is None


def test_letter_takes_priority_over_other_rules():
    assert classify_intent_rules("a").intent == "answer"


def _question() -> CurrentQuestion:
    return CurrentQuestion(
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
    )


def test_aclassify_uses_rules_first_and_skips_llm():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=AssertionError("LLM should not be called"))
    result = asyncio.run(aclassify_intent("A", _question(), llm))
    assert result.intent == "answer"
    assert result.answer_index == "A"
    llm.ainvoke.assert_not_called()


def test_aclassify_llm_returns_answer_with_index():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=json.dumps({
        "intent": "answer", "answer_index": "B", "language_match": True,
    }))
    result = asyncio.run(aclassify_intent("loss or undesirable", _question(), llm))
    assert result.intent == "answer"
    assert result.answer_index == "B"


def test_aclassify_language_mismatch_forces_question():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=json.dumps({
        "intent": "answer", "answer_index": "B", "language_match": False,
    }))
    result = asyncio.run(aclassify_intent("đáp án là loss", _question(), llm))
    assert result.intent == "question"
    assert result.answer_index is None


def test_aclassify_llm_returns_question():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=json.dumps({
        "intent": "question", "answer_index": None, "language_match": True,
    }))
    result = asyncio.run(aclassify_intent("can you explain risk?", _question(), llm))
    assert result.intent == "question"
    assert result.answer_index is None


def test_aclassify_llm_returns_answer_with_null_index():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=json.dumps({
        "intent": "answer", "answer_index": None, "language_match": True,
    }))
    result = asyncio.run(aclassify_intent("hmm not sure", _question(), llm))
    assert result.intent == "answer"
    assert result.answer_index is None


def test_aclassify_llm_failure_falls_back_to_question():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))
    result = asyncio.run(aclassify_intent("random text", _question(), llm))
    assert result.intent == "question"
