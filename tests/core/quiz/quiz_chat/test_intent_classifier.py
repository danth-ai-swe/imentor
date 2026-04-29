import pytest

from src.core.quiz.quiz_chat.intent_classifier import (
    IntentResult,
    classify_intent_rules,
)


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
