"""Quiz chat intent classifier — rule-based fast path with rapidfuzz."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from rapidfuzz import fuzz, process

from src.core.quiz.quiz_chat.keywords import (
    FINISH_KEYWORDS,
    HINT_KEYWORDS,
    normalise,
)
from src.core.quiz.quiz_chat.model import CurrentQuestion
from src.core.quiz.quiz_chat.prompts import CLASSIFY_INTENT_PROMPT
from src.rag.llm.chat_llm import AzureChatClient
from src.utils.app_utils import parse_json_response
from src.utils.logger_utils import logger

Intent = Literal["hint", "finish", "answer", "question"]
AnswerIndex = Literal["A", "B", "C", "D"]

FUZZY_THRESHOLD = 85
# Below this length, a keyword can only match exactly. Short keywords
# (e.g. "hin", "dum") would otherwise fuzzy-match unrelated tokens with
# only one different letter and trip false positives.
MIN_FUZZY_LEN = 5
DIGIT_TO_INDEX: dict[str, AnswerIndex] = {"1": "A", "2": "B", "3": "C", "4": "D"}

_LETTER_RE = re.compile(r"^[abcd]$")
_DIGIT_RE = re.compile(r"^[1-4]$")


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    answer_index: Optional[AnswerIndex] = None


def _fuzzy_in(text: str, choices: frozenset[str]) -> bool:
    if text in choices:
        return True
    if len(text) < MIN_FUZZY_LEN:
        return False
    match = process.extractOne(text, choices, scorer=fuzz.ratio)
    return match is not None and match[1] >= FUZZY_THRESHOLD


def classify_intent_rules(message: str) -> Optional[IntentResult]:
    """Return an IntentResult if a deterministic rule matches, else None."""
    text = normalise(message)
    if not text:
        return None

    if _LETTER_RE.match(text):
        return IntentResult(intent="answer", answer_index=text.upper())  # type: ignore[arg-type]

    if _DIGIT_RE.match(text):
        return IntentResult(intent="answer", answer_index=DIGIT_TO_INDEX[text])

    if _fuzzy_in(text, FINISH_KEYWORDS):
        return IntentResult(intent="finish")

    if _fuzzy_in(text, HINT_KEYWORDS):
        return IntentResult(intent="hint")

    return None


def _format_options_block(question: CurrentQuestion) -> str:
    return "\n".join(f"{opt.index}. {opt.text}" for opt in question.options)


async def aclassify_intent(
    message: str,
    current_question: CurrentQuestion,
    llm: AzureChatClient,
) -> IntentResult:
    """Classify intent: rule-based first, LLM fallback. Always returns an IntentResult."""
    rule_hit = classify_intent_rules(message)
    if rule_hit is not None:
        return rule_hit

    prompt = CLASSIFY_INTENT_PROMPT.format(
        question_text=current_question.question,
        options_block=_format_options_block(current_question),
        message=message,
    )
    try:
        raw = (await llm.ainvoke(prompt)).strip()
        parsed = parse_json_response(raw)
    except Exception:
        logger.exception("Quiz-chat LLM classification failed; defaulting to question intent")
        return IntentResult(intent="question")

    intent = parsed.get("intent", "question")
    answer_index = parsed.get("answer_index")
    language_match = parsed.get("language_match", True)

    if intent == "answer" and not language_match:
        return IntentResult(intent="question")

    if intent == "answer":
        if answer_index in ("A", "B", "C", "D"):
            return IntentResult(intent="answer", answer_index=answer_index)
        if answer_index is not None:
            logger.warning(
                "Quiz-chat LLM returned invalid answer_index=%r; coercing to null",
                answer_index,
            )
        return IntentResult(intent="answer", answer_index=None)

    return IntentResult(intent="question")
