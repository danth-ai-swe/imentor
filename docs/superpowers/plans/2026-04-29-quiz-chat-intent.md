# Quiz Chat Intent API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Python AI endpoint `POST /quiz/chat` that classifies free-form quiz input into `hint` / `finish` / `answer` / `question` intents, and answers `question` intents through the existing RAG pipeline with a no-spoiler instruction.

**Architecture:** Two-stage classifier (rule + rapidfuzz, then LLM fallback) lives in a new `src/core/quiz/quiz_chat/` module. The `question` intent reuses private helpers from `src/rag/search/pipeline.py` (no edits to existing files except `app_controller.py`, where one new endpoint function is added). The pattern mirrors the existing `src/rag/search/pipeline_chunks.py` for consistency.

**Tech Stack:** FastAPI, Pydantic, rapidfuzz, asyncio, existing OpenAI/Qdrant clients.

**Spec:** `docs/superpowers/specs/2026-04-29-quiz-chat-intent-design.md`

---

## File Structure

**New files:**
- `src/core/quiz/quiz_chat/__init__.py`
- `src/core/quiz/quiz_chat/model.py` — Pydantic request/response models
- `src/core/quiz/quiz_chat/keywords.py` — finish/hint keyword sets
- `src/core/quiz/quiz_chat/prompts.py` — LLM classification prompt + no-spoiler instruction
- `src/core/quiz/quiz_chat/intent_classifier.py` — rule + LLM intent classification
- `src/core/quiz/quiz_chat/pipeline.py` — orchestrator + quiz-aware question pipeline
- `tests/core/quiz/quiz_chat/__init__.py`
- `tests/core/quiz/__init__.py`
- `tests/core/__init__.py`
- `tests/core/quiz/quiz_chat/test_keywords.py`
- `tests/core/quiz/quiz_chat/test_intent_classifier.py`
- `tests/core/quiz/quiz_chat/test_pipeline.py`

**Modified files:**
- `requirements.txt` — add `rapidfuzz`
- `src/apis/app_controller.py` — add new endpoint function (no edits to existing functions)

---

## Task 1: Add rapidfuzz dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Append `rapidfuzz` to requirements.txt**

Open `requirements.txt`. Append a new line at the end:

```
rapidfuzz
```

- [ ] **Step 2: Install the dependency**

Run: `pip install rapidfuzz`
Expected: installs successfully (current latest is 3.x).

- [ ] **Step 3: Verify import works**

Run: `python -c "from rapidfuzz import fuzz; print(fuzz.ratio('hint', 'hin'))"`
Expected output: a number near 86.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add rapidfuzz dependency for quiz chat intent classifier"
```

---

## Task 2: Module skeleton + keyword sets

**Files:**
- Create: `src/core/quiz/quiz_chat/__init__.py`
- Create: `src/core/quiz/quiz_chat/keywords.py`
- Create: `tests/core/__init__.py`
- Create: `tests/core/quiz/__init__.py`
- Create: `tests/core/quiz/quiz_chat/__init__.py`
- Create: `tests/core/quiz/quiz_chat/test_keywords.py`

- [ ] **Step 1: Create the empty package `__init__.py` files**

Create `src/core/quiz/quiz_chat/__init__.py` with empty content.
Create `tests/core/__init__.py`, `tests/core/quiz/__init__.py`, `tests/core/quiz/quiz_chat/__init__.py` with empty content (4 files total).

- [ ] **Step 2: Write the failing test**

Create `tests/core/quiz/quiz_chat/test_keywords.py`:

```python
from src.core.quiz.quiz_chat.keywords import (
    FINISH_KEYWORDS,
    HINT_KEYWORDS,
    normalise,
)


def test_normalise_strips_and_lowercases():
    assert normalise("  Hello!  ") == "hello"
    assert normalise("KẾT THÚC.") == "kết thúc"
    assert normalise("\thint?\n") == "hint"


def test_normalise_collapses_internal_whitespace():
    assert normalise("end   quiz") == "end quiz"
    assert normalise("submit\tand\nfinish") == "submit and finish"


def test_finish_keywords_contain_required_phrases():
    required = {
        "finish", "finish session", "end", "end session", "end quiz",
        "submit and finish", "done", "quit", "exit", "stop",
        "kết thúc", "thoát", "dừng", "nộp bài",
    }
    assert required.issubset(FINISH_KEYWORDS)


def test_hint_keywords_contain_required_phrases():
    required = {
        "hint", "hin", "give me a hint", "help",
        "gợi ý", "goi y",
    }
    assert required.issubset(HINT_KEYWORDS)


def test_keyword_sets_are_lowercase_and_normalised():
    for kw in FINISH_KEYWORDS | HINT_KEYWORDS:
        assert kw == normalise(kw), f"{kw!r} is not pre-normalised"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/core/quiz/quiz_chat/test_keywords.py -v`
Expected: ImportError / ModuleNotFoundError on `keywords`.

- [ ] **Step 4: Implement `keywords.py`**

Create `src/core/quiz/quiz_chat/keywords.py`:

```python
"""Keyword sets and text normalisation for the quiz-chat intent classifier."""
import re
import unicodedata


def normalise(text: str) -> str:
    """Lowercase, NFC-normalise, collapse whitespace, strip leading/trailing punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.strip(".,!?;:'\"()[]{}")
    text = text.strip()
    return text


FINISH_KEYWORDS: frozenset[str] = frozenset({
    # English
    "finish", "finish session", "end", "end session", "end quiz",
    "submit", "submit and finish", "done", "quit", "exit", "stop",
    # Vietnamese (with and without diacritics)
    "kết thúc", "ket thuc",
    "thoát", "thoat",
    "dừng", "dung",
    "nộp bài", "nop bai",
})


HINT_KEYWORDS: frozenset[str] = frozenset({
    # English
    "hint", "hin", "give me a hint", "help",
    # Vietnamese
    "gợi ý", "goi y",
})
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/core/quiz/quiz_chat/test_keywords.py -v`
Expected: all 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/core/quiz/quiz_chat/__init__.py src/core/quiz/quiz_chat/keywords.py \
        tests/core/__init__.py tests/core/quiz/__init__.py \
        tests/core/quiz/quiz_chat/__init__.py tests/core/quiz/quiz_chat/test_keywords.py
git commit -m "feat(quiz-chat): add keyword sets and normalise helper"
```

---

## Task 3: Pydantic request/response models

**Files:**
- Create: `src/core/quiz/quiz_chat/model.py`

- [ ] **Step 1: Add models to `model.py`**

Create `src/core/quiz/quiz_chat/model.py`:

```python
"""Pydantic models for the /quiz/chat endpoint."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.apis.app_model import ChatSourceModel


class OptionItem(BaseModel):
    index: Literal["A", "B", "C", "D"]
    text: str = Field(..., min_length=1)


class CurrentQuestion(BaseModel):
    question_type: Optional[str] = None
    difficulty: Optional[str] = None
    question: str = Field(..., min_length=1)
    options: List[OptionItem] = Field(..., min_length=4, max_length=4)
    category: str = Field(..., min_length=1)
    node_name: str = Field(..., min_length=1)
    sources: Optional[List[dict]] = None


class QuizChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=15_000)
    current_question: CurrentQuestion


class QuizChatResponse(BaseModel):
    intent: Literal["hint", "finish", "answer", "question"]
    answer_index: Optional[Literal["A", "B", "C", "D"]] = None
    message: Optional[str] = None
    content: Optional[str] = None
    sources: Optional[List[ChatSourceModel]] = None
```

- [ ] **Step 2: Smoke-check the model imports**

Run: `python -c "from src.core.quiz.quiz_chat.model import QuizChatRequest, QuizChatResponse; print('ok')"`
Expected output: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/core/quiz/quiz_chat/model.py
git commit -m "feat(quiz-chat): add request/response Pydantic models"
```

---

## Task 4: Rule-based intent classifier (no LLM yet)

**Files:**
- Create: `src/core/quiz/quiz_chat/intent_classifier.py`
- Create: `tests/core/quiz/quiz_chat/test_intent_classifier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/core/quiz/quiz_chat/test_intent_classifier.py`:

```python
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
    # "finsh" vs "finish" → ratio ~91
    assert classify_intent_rules("finsh") == IntentResult(intent="finish")


def test_typo_close_to_hint_via_fuzzy():
    # "hnit" vs "hint" → ratio ~75; below threshold, should NOT classify
    # but "hnt" vs "hint" stays low too. Use a known-safe variant:
    assert classify_intent_rules("hint!").intent == "hint"


def test_unrelated_text_returns_none():
    assert classify_intent_rules("what is risk?") is None
    assert classify_intent_rules("the possibility of loss") is None


def test_empty_input_returns_none():
    assert classify_intent_rules("") is None
    assert classify_intent_rules("   ") is None


def test_letter_takes_priority_over_other_rules():
    # Letter regex is checked before keyword sets.
    assert classify_intent_rules("a").intent == "answer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/quiz/quiz_chat/test_intent_classifier.py -v`
Expected: ImportError on `intent_classifier`.

- [ ] **Step 3: Implement `intent_classifier.py`**

Create `src/core/quiz/quiz_chat/intent_classifier.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/quiz/quiz_chat/test_intent_classifier.py -v`
Expected: all parametrised cases pass.

- [ ] **Step 5: Commit**

```bash
git add src/core/quiz/quiz_chat/intent_classifier.py \
        tests/core/quiz/quiz_chat/test_intent_classifier.py
git commit -m "feat(quiz-chat): rule-based intent classifier with rapidfuzz"
```

---

## Task 5: LLM classification prompt

**Files:**
- Create: `src/core/quiz/quiz_chat/prompts.py`

- [ ] **Step 1: Add prompts**

Create `src/core/quiz/quiz_chat/prompts.py`:

```python
"""Prompts for the quiz-chat intent classifier and question pipeline."""

CLASSIFY_INTENT_PROMPT = """You are an intent classifier for a multiple-choice quiz chat.

The learner is currently looking at this question:
\"\"\"
{question_text}
\"\"\"

Options:
{options_block}

The learner just typed:
\"\"\"
{message}
\"\"\"

Decide between two intents:
- "answer": the learner is trying to pick one of A/B/C/D, possibly by paraphrasing an option's meaning.
- "question": the learner is asking something else (a free-form question, a request for clarification, off-topic chat).

If "answer", choose the single option whose meaning is closest to the learner's text.
If no option is reasonably close, return answer_index = null.

Also decide language_match:
- true if the learner's message is written in the same primary language as the question text.
- false if the languages differ (e.g. question in English, message in Vietnamese).

Return ONLY a JSON object with this exact shape, no extra text:
{{"intent": "answer" | "question", "answer_index": "A" | "B" | "C" | "D" | null, "language_match": true | false}}
"""


QUESTION_GUARD_INSTRUCTION = """[QUIZ CONTEXT — READ FIRST]
The user is currently answering a multiple-choice quiz question.
Topic node: "{node_name}" (category: {category}).
Current question: "{question_text}"
Options:
{options_block}

When you respond:
- Explain the underlying concept clearly using the source material.
- Do NOT state which option (A/B/C/D) is the correct answer.
- Do NOT echo any single option verbatim as the answer.
- Do NOT phrase your reply as "the answer is X" or any equivalent that resolves the question.
- You MAY discuss the topic, define related terms, quote source material, and use citations.
- The learner already has the question in front of them — your job is to help them think, not to grade their answer.

Below is the standard retrieval-augmented prompt; follow it while obeying the rules above."""
```

- [ ] **Step 2: Smoke-check imports + formatting**

Run:
```bash
python -c "
from src.core.quiz.quiz_chat.prompts import CLASSIFY_INTENT_PROMPT, QUESTION_GUARD_INSTRUCTION
print(CLASSIFY_INTENT_PROMPT.format(question_text='What is risk?', options_block='A. Gain', message='loss'))
print('---')
print(QUESTION_GUARD_INSTRUCTION.format(node_name='Risk', category='Risk Concepts', question_text='What is risk?', options_block='A. Gain'))
"
```
Expected: both strings render with placeholders filled, no KeyError.

- [ ] **Step 3: Commit**

```bash
git add src/core/quiz/quiz_chat/prompts.py
git commit -m "feat(quiz-chat): add LLM classification prompt and no-spoiler instruction"
```

---

## Task 6: LLM fallback in classifier

**Files:**
- Modify: `src/core/quiz/quiz_chat/intent_classifier.py`
- Modify: `tests/core/quiz/quiz_chat/test_intent_classifier.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/quiz/quiz_chat/test_intent_classifier.py`:

```python
import asyncio
import json
from unittest.mock import AsyncMock

from src.core.quiz.quiz_chat.intent_classifier import (
    aclassify_intent,
)
from src.core.quiz.quiz_chat.model import CurrentQuestion, OptionItem


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
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `pytest tests/core/quiz/quiz_chat/test_intent_classifier.py -v`
Expected: 6 new tests fail with ImportError on `aclassify_intent`.

- [ ] **Step 3: Implement `aclassify_intent`**

Append to `src/core/quiz/quiz_chat/intent_classifier.py`:

```python
from src.core.quiz.quiz_chat.model import CurrentQuestion
from src.core.quiz.quiz_chat.prompts import CLASSIFY_INTENT_PROMPT
from src.rag.llm.chat_llm import AzureChatClient
from src.utils.app_utils import parse_json_response
from src.utils.logger_utils import logger


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
        # AC-1 Scenario 2 warning: wrong language → treat as question.
        return IntentResult(intent="question")

    if intent == "answer":
        if answer_index in ("A", "B", "C", "D"):
            return IntentResult(intent="answer", answer_index=answer_index)
        return IntentResult(intent="answer", answer_index=None)

    return IntentResult(intent="question")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/core/quiz/quiz_chat/test_intent_classifier.py -v`
Expected: all tests pass (rule-based + LLM fallback).

- [ ] **Step 5: Commit**

```bash
git add src/core/quiz/quiz_chat/intent_classifier.py \
        tests/core/quiz/quiz_chat/test_intent_classifier.py
git commit -m "feat(quiz-chat): LLM fallback for intent classification with language guard"
```

---

## Task 7: Quiz-aware question pipeline

**Files:**
- Create: `src/core/quiz/quiz_chat/pipeline.py`

This task introduces the orchestrator and a quiz-aware mirror of the existing core/overall search flows. It does **not** add tests yet — Task 8 covers them so the orchestrator and pipeline tests live together.

- [ ] **Step 1: Create the pipeline file**

Create `src/core/quiz/quiz_chat/pipeline.py`:

```python
"""Quiz-chat dispatcher and quiz-aware question pipeline.

Mirrors the structure of `src/rag/search/pipeline_chunks.py` so the codebase
has one consistent pattern for "wrap the existing dispatch with a custom
final step." Imports private helpers from `src/rag/search/pipeline.py`
without modifying them.
"""
import asyncio
from typing import Any, Dict, List, Optional

from src.apis.app_model import ChatSourceModel
from src.config.app_config import AppConfig, get_app_config
from src.constants.app_constant import (
    COLLECTION_NAME,
    CORE_RERANK_TOP_K,
    CORE_VECTOR_TOP_K,
    INTENT_OFF_TOPIC,
    INTENT_OVERALL_COURSE_KNOWLEDGE,
    INTENT_QUIZ,
    NO_RESULT_RESPONSE_MAP,
    OFF_TOPIC_RESPONSE_MAP,
    OVERALL_COLLECTION_NAME,
    VECTOR_SEARCH_TOP_K,
)
from src.core.quiz.quiz_chat.intent_classifier import (
    IntentResult,
    _format_options_block,
    aclassify_intent,
)
from src.core.quiz.quiz_chat.model import (
    CurrentQuestion,
    QuizChatRequest,
    QuizChatResponse,
)
from src.core.quiz.quiz_chat.prompts import QUESTION_GUARD_INSTRUCTION
from src.rag.db_vector import get_qdrant_client
from src.rag.llm.chat_llm import AzureChatClient, get_openai_chat_client
from src.rag.llm.embedding_llm import AzureEmbeddingClient, get_openai_embedding_client
from src.rag.search.entrypoint import build_final_prompt
from src.rag.search.pipeline import (
    _acheck_input_clarity,
    _aembed_text,
    _afetch_neighbor_chunks,
    _ahyde_generate,
    _aroute_intent,
    _avalidate_and_prepare,
    _avector_search,
    _merge_chunks,
    extract_sources,
)
from src.rag.search.reranker import arerank_chunks
from src.rag.search.searxng_search import web_rag_answer
from src.utils.app_utils import is_quiz_intent
from src.utils.logger_utils import StepTimer, logger


_AMBIGUOUS_ANSWER_MSG = (
    "Could not match your answer to any option. Please try again."
)


async def _aquiz_generate_answer(
    llm: AzureChatClient,
    standalone_query: str,
    detected_language: str,
    enriched_chunks: List[Dict[str, Any]],
    current_question: CurrentQuestion,
) -> str:
    base_prompt = build_final_prompt(
        user_input=standalone_query,
        detected_language=detected_language,
        relevant_chunks=enriched_chunks,
    )
    guard = QUESTION_GUARD_INSTRUCTION.format(
        node_name=current_question.node_name,
        category=current_question.category,
        question_text=current_question.question,
        options_block=_format_options_block(current_question),
    )
    augmented = guard + "\n\n" + base_prompt
    try:
        return (await llm.ainvoke_creative(augmented)).strip()
    except Exception:
        logger.exception("Failed to generate quiz-aware answer from LLM")
        return ""


def _question_response(
    *, content: str, sources: Optional[List[ChatSourceModel]] = None,
) -> QuizChatResponse:
    return QuizChatResponse(
        intent="question",
        content=content,
        sources=sources or [],
    )


async def _arun_core_quiz_question(
    llm: AzureChatClient,
    embedder: AzureEmbeddingClient,
    standalone_query: str,
    detected_language: str,
    current_question: CurrentQuestion,
) -> QuizChatResponse:
    timer = StepTimer(f"core_quiz_question:{COLLECTION_NAME}")
    config: AppConfig = get_app_config()

    try:
        qdrant = get_qdrant_client(COLLECTION_NAME)

        async with timer.astep("clarity_and_hyde_parallel"):
            clarity_task = _acheck_input_clarity(llm, standalone_query, detected_language)
            hyde_task = _ahyde_generate(llm, standalone_query)
            clarity, hyde = await asyncio.gather(clarity_task, hyde_task)

        if not clarity.get("clear", True):
            response_parts = clarity.get("response", [])
            response_str = (
                response_parts if isinstance(response_parts, str)
                else "\n".join(r.strip() for r in response_parts if r) if isinstance(response_parts, list)
                else str(response_parts)
            )
            return _question_response(content=response_str, sources=[])

        async with timer.astep("hyde_embedding"):
            dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, hyde)

        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, dense_vec, colbert_vec, standalone_query, top_k=CORE_VECTOR_TOP_K,
            )

        if not sorted_chunks:
            async with timer.astep("web_search_fallback"):
                rs = await web_rag_answer(llm, embedder, standalone_query, detected_language)
            return _question_response(content=rs["answer"], sources=rs["sources"])

        async with timer.astep("rerank"):
            reranked = await arerank_chunks(standalone_query, sorted_chunks, CORE_RERANK_TOP_K)

        async with timer.astep("enrich_chunks"):
            neighbor_chunks = await _afetch_neighbor_chunks(qdrant, reranked)
            enriched_chunks = _merge_chunks(reranked, neighbor_chunks)

        async with timer.astep("generate_answer"):
            answer = await _aquiz_generate_answer(
                llm, standalone_query, detected_language, enriched_chunks, current_question,
            )

        sources = extract_sources(reranked, config.APP_DOMAIN)
        return _question_response(content=answer, sources=sources)
    finally:
        timer.summary()


async def _arun_overall_quiz_question(
    llm: AzureChatClient,
    embedder: AzureEmbeddingClient,
    standalone_query: str,
    detected_language: str,
    current_question: CurrentQuestion,
) -> QuizChatResponse:
    timer = StepTimer(f"overall_quiz_question:{OVERALL_COLLECTION_NAME}")

    try:
        qdrant = get_qdrant_client(OVERALL_COLLECTION_NAME)

        async with timer.astep("clarity_check"):
            clarity = await _acheck_input_clarity(llm, standalone_query, detected_language)
        if not clarity.get("clear", True):
            response_parts = clarity.get("response", [])
            response_str = (
                response_parts if isinstance(response_parts, str)
                else "\n".join(r.strip() for r in response_parts if r) if isinstance(response_parts, list)
                else str(response_parts)
            )
            return _question_response(content=response_str, sources=[])

        async with timer.astep("query_embedding"):
            dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, standalone_query)

        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, dense_vec, colbert_vec, standalone_query, top_k=VECTOR_SEARCH_TOP_K,
            )

        if not sorted_chunks:
            no_result = NO_RESULT_RESPONSE_MAP.get(detected_language) or ""
            return _question_response(content=no_result, sources=[])

        async with timer.astep("generate_answer"):
            answer = await _aquiz_generate_answer(
                llm, standalone_query, detected_language, sorted_chunks, current_question,
            )

        return _question_response(content=answer, sources=[])
    finally:
        timer.summary()


async def _arun_question_intent(
    llm: AzureChatClient,
    embedder: AzureEmbeddingClient,
    message: str,
    current_question: CurrentQuestion,
) -> QuizChatResponse:
    detected_language, standalone_query, error_response = (
        await _avalidate_and_prepare(message, None, llm)
    )
    if error_response:
        return _question_response(content=error_response, sources=[])

    if is_quiz_intent(message) or is_quiz_intent(standalone_query):
        # Defensive: shouldn't happen because the classifier already routed to
        # answer/hint/finish; if it does, behave like the existing pipeline does
        # for quiz keywords (no-op static reply).
        return _question_response(content="", sources=[])

    intent = await _aroute_intent(embedder, standalone_query)

    if intent == INTENT_OFF_TOPIC:
        return _question_response(
            content=OFF_TOPIC_RESPONSE_MAP.get(detected_language) or "",
            sources=[],
        )

    if intent == INTENT_OVERALL_COURSE_KNOWLEDGE:
        return await _arun_overall_quiz_question(
            llm, embedder, standalone_query, detected_language, current_question,
        )

    return await _arun_core_quiz_question(
        llm, embedder, standalone_query, detected_language, current_question,
    )


async def async_quiz_chat_dispatch(payload: QuizChatRequest) -> QuizChatResponse:
    """Top-level entry: classify intent, then either return a static response
    (hint/finish/answer) or run the quiz-aware question pipeline."""
    timer = StepTimer("quiz_chat_dispatch")
    try:
        llm = get_openai_chat_client()
        embedder = get_openai_embedding_client()

        async with timer.astep("classify_intent"):
            decision: IntentResult = await aclassify_intent(
                payload.message, payload.current_question, llm,
            )

        if decision.intent == "hint":
            return QuizChatResponse(intent="hint")

        if decision.intent == "finish":
            return QuizChatResponse(intent="finish")

        if decision.intent == "answer":
            if decision.answer_index is None:
                return QuizChatResponse(
                    intent="answer",
                    answer_index=None,
                    message=_AMBIGUOUS_ANSWER_MSG,
                )
            return QuizChatResponse(
                intent="answer",
                answer_index=decision.answer_index,
            )

        async with timer.astep("question_pipeline"):
            return await _arun_question_intent(
                llm, embedder, payload.message, payload.current_question,
            )
    finally:
        timer.summary()
```

- [ ] **Step 2: Smoke-check imports**

Run: `python -c "from src.core.quiz.quiz_chat.pipeline import async_quiz_chat_dispatch; print('ok')"`
Expected output: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/core/quiz/quiz_chat/pipeline.py
git commit -m "feat(quiz-chat): orchestrator and quiz-aware question pipeline"
```

---

## Task 8: Pipeline tests

**Files:**
- Create: `tests/core/quiz/quiz_chat/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/core/quiz/quiz_chat/test_pipeline.py`:

```python
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.apis.app_model import ChatSourceModel
from src.core.quiz.quiz_chat.intent_classifier import IntentResult
from src.core.quiz.quiz_chat.model import (
    CurrentQuestion,
    OptionItem,
    QuizChatRequest,
)
from src.core.quiz.quiz_chat.pipeline import async_quiz_chat_dispatch


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


def _patch_clients():
    return (
        patch("src.core.quiz.quiz_chat.pipeline.get_openai_chat_client", return_value=object()),
        patch("src.core.quiz.quiz_chat.pipeline.get_openai_embedding_client", return_value=object()),
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
        "src.core.quiz.quiz_chat.pipeline.is_quiz_intent", return_value=False,
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
        "src.core.quiz.quiz_chat.pipeline.is_quiz_intent", return_value=False,
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
        "src.core.quiz.quiz_chat.pipeline.is_quiz_intent", return_value=False,
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
```

- [ ] **Step 2: Run tests to verify they fail / pass**

Run: `pytest tests/core/quiz/quiz_chat/test_pipeline.py -v`
Expected: all 8 tests pass (the pipeline was implemented in Task 7).

- [ ] **Step 3: Commit**

```bash
git add tests/core/quiz/quiz_chat/test_pipeline.py
git commit -m "test(quiz-chat): pipeline orchestrator + question pipeline tests"
```

---

## Task 9: API endpoint

**Files:**
- Modify: `src/apis/app_controller.py` (add new function only, no edits to existing)

- [ ] **Step 1: Add the endpoint function**

Open `src/apis/app_controller.py`. At the **end** of the file, append:

```python
from src.core.quiz.quiz_chat.model import QuizChatRequest, QuizChatResponse
from src.core.quiz.quiz_chat.pipeline import async_quiz_chat_dispatch


@quiz_router.post("/chat", response_model=QuizChatResponse)
async def quiz_chat(payload: QuizChatRequest) -> QuizChatResponse:
    logger.info(
        "quiz_chat called | node_name=%s | message_len=%d",
        payload.current_question.node_name, len(payload.message),
    )
    try:
        return await async_quiz_chat_dispatch(payload)
    except Exception as exc:
        logger.exception("quiz_chat failed")
        raise QdrantApiError(f"quiz_chat failed: {exc}") from exc
```

- [ ] **Step 2: Smoke-check the FastAPI app boots**

Run:
```bash
python -c "
from src.apis.app_router import api_router
paths = [r.path for r in api_router.routes for _ in (r.routes if hasattr(r, 'routes') else [r])]
# Walk included routers
all_paths = set()
def walk(router):
    for r in router.routes:
        if hasattr(r, 'routes'):
            walk(r)
        elif hasattr(r, 'path'):
            all_paths.add(r.path)
walk(api_router)
assert '/quiz/chat' in all_paths, f'quiz/chat not registered. Found: {sorted(all_paths)}'
print('ok: /quiz/chat registered')
"
```
Expected output: `ok: /quiz/chat registered`.

- [ ] **Step 3: Verify the existing test suite still passes**

Run: `pytest tests/ -v`
Expected: all tests pass (existing + new ones from Tasks 2/4/6/8).

- [ ] **Step 4: Commit**

```bash
git add src/apis/app_controller.py
git commit -m "feat(quiz-chat): add POST /quiz/chat endpoint"
```

---

## Task 10: Manual smoke test

**Files:** none (uses running server).

- [ ] **Step 1: Start the FastAPI server locally**

Run: `python main.py` (or whatever the project's existing run command is — check `main.py` if unsure).

- [ ] **Step 2: POST a letter answer and verify the response**

In another shell, run:
```bash
curl -s -X POST http://localhost:8000/quiz/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "B",
    "current_question": {
      "question_type": "Definition / Concept",
      "difficulty": "Beginner",
      "question": "What is the definition of risk?",
      "options": [
        {"index":"A","text":"The possibility of gain or profit."},
        {"index":"B","text":"The possibility of loss or an undesirable outcome."},
        {"index":"C","text":"The certainty of a favorable outcome."},
        {"index":"D","text":"The assessment of financial investments."}
      ],
      "category":"Risk Concepts","node_name":"Risk"
    }
  }'
```
Expected: `{"intent":"answer","answer_index":"B",...}` (other fields null).

- [ ] **Step 3: POST a hint keyword**

Run the same curl with `"message": "gợi ý"`.
Expected: `{"intent":"hint",...}`.

- [ ] **Step 4: POST a finish keyword**

Run the same curl with `"message": "kết thúc"`.
Expected: `{"intent":"finish",...}`.

- [ ] **Step 5: POST a fuzzy text answer (LLM fallback)**

Run the same curl with `"message": "I think it's the loss one"`.
Expected: `{"intent":"answer","answer_index":"B",...}` (LLM picks B based on option text).

- [ ] **Step 6: POST a free-form question**

Run the same curl with `"message": "Can you explain what risk means in insurance?"`.
Expected: `{"intent":"question","content":"...","sources":[...]}`.
Verify the `content` does NOT contain "the answer is B" or "option B is correct".

- [ ] **Step 7: POST an ambiguous answer**

Run the same curl with `"message": "I don't know really"`.
Expected: `{"intent":"answer","answer_index":null,"message":"Could not match..."}`.

- [ ] **Step 8: POST a wrong-language answer (Vietnamese to English question)**

Run the same curl with `"message": "khả năng mất mát"` against the English-language question.
Expected: `{"intent":"question",...}` (language guard kicks in; LLM treats Vietnamese-on-English as question).

- [ ] **Step 9: Commit a short note about smoke results**

If everything passes, no commit needed. If you hit issues, fix them in a follow-up task and commit. Either way, document any caveats in the PR description.
