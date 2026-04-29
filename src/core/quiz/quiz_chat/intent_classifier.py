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

Intent = Literal["hint", "finish", "answer", "question"]
AnswerIndex = Literal["A", "B", "C", "D"]

FUZZY_THRESHOLD = 85
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
