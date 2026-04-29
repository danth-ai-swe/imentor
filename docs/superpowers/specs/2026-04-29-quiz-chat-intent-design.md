# Quiz Chat Intent API — Design

**Date:** 2026-04-29
**Branch:** `feat/java-rabbitmq-react-chat`
**Author:** danth-ai-swe (paired with Claude)

## 1. Problem statement

The quiz UI lets the learner answer multiple-choice questions by clicking buttons (A/B/C/D, Hint, Finish Session). The product wants to also let the learner **type freely into the chat input** during a quiz. The Python AI service must understand what the typed message *means* and tell the Java backend which action to trigger.

Four intents must be recognised from free-form text:

1. **answer** — pick option A/B/C/D for the current question.
2. **hint** — request a hint (same as clicking the Hint button).
3. **finish** — end the quiz session (same as clicking Finish Session).
4. **question** — a free-form question; falls back to the existing RAG pipeline.

Recognition must work across formats and languages:

| Pattern | Example inputs |
|---|---|
| Letter | `A`, `b`, `C` |
| Index | `1`, `2`, `4` |
| Fuzzy text matching an option | `"loss or undesirable outcome"` for option B |
| Finish keyword | `finish`, `end quiz`, `done`, `exit`, `kết thúc`, `nộp bài`, `thoát` |
| Hint keyword | `hint`, `hin`, `gợi ý`, `goi y`, `help`, `give me a hint` |

Constraints:

- **No history** in the question pipeline. Each quiz prompt is independent.
- **No spoiler**: the AI's free-form answer must not directly reveal which option is correct.
- **Stateless API**: the FE passes the current question object on every call.
- **Don't break existing code**: this is a strictly additive change. No edits to the existing dispatch pipeline.

## 2. Scope

In scope (this task):
- New Python AI endpoint `POST /quiz/chat`.
- New module `src/core/quiz/quiz_chat/` with intent classifier and pipeline.
- New request/response models.
- Reuse private helpers from `src/rag/search/pipeline.py` for the `question` intent.

Out of scope (separate tasks, do **not** touch in this PR):
- Java backend (will adapt the new endpoint and enforce business rules: hint-already-shown, must-answer-before-finish).
- React frontend (will route typed input through the new endpoint).
- Existing chat endpoints and pipeline functions.

## 3. API contract

### 3.1 Endpoint

```
POST /quiz/chat
Content-Type: application/json
```

### 3.2 Request

```json
{
  "message": "I think it's about loss",
  "current_question": {
    "question_type": "Definition / Concept",
    "difficulty": "Beginner",
    "question": "What is the definition of risk?",
    "options": [
      { "index": "A", "text": "The possibility of gain or profit." },
      { "index": "B", "text": "The possibility of loss or an undesirable outcome." },
      { "index": "C", "text": "The certainty of a favorable outcome." },
      { "index": "D", "text": "The assessment of financial investments." }
    ],
    "category": "Risk Concepts",
    "node_name": "Risk",
    "sources": [ /* original quiz sources, not used for classification */ ]
  }
}
```

Validation:
- `message`: 1–15,000 chars (matches existing `ChatRequest`).
- `current_question.options`: exactly 4 items with index `A|B|C|D`.
- `current_question.node_name` and `category`: non-empty strings (used for question-intent prompt context).
- `correct_answer` is intentionally **not** part of the schema. The Java backend keeps it private to avoid leaking it to the model.

### 3.3 Response shapes

The response always carries `intent`. Other fields depend on the intent.

```json
// hint
{ "intent": "hint" }

// finish
{ "intent": "finish" }

// answer — index resolved
{ "intent": "answer", "answer_index": "B" }

// answer — could not resolve (ambiguous text, no option matched)
{
  "intent": "answer",
  "answer_index": null,
  "message": "Could not match your answer to any option. Please try again."
}

// question — free-form RAG
{
  "intent": "question",
  "content": "Risk in insurance refers to the chance that an undesirable event will occur...",
  "sources": [
    {
      "name": "LOMA281_M1L1_...",
      "url": "https://.../file.pdf",
      "page_number": 1,
      "total_pages": 31
    }
  ]
}
```

The response model is a single Pydantic class `QuizChatResponse` with optional fields; only the fields relevant to the resolved intent are populated.

## 4. Module layout

```
src/core/quiz/quiz_chat/
├── __init__.py
├── model.py              # QuizChatRequest, QuizChatResponse, CurrentQuestion, OptionItem
├── intent_classifier.py  # classify_intent(message, options) → IntentResult
├── prompts.py            # CLASSIFY_INTENT_PROMPT, QUESTION_GUARD_INSTRUCTION
├── keywords.py           # FINISH_KEYWORDS, HINT_KEYWORDS (multi-language sets)
└── pipeline.py           # async_quiz_chat_dispatch — orchestrator
```

Endpoint registration: a new function `quiz_chat` added to `src/apis/app_controller.py`, attached to the existing `quiz_router` (`/quiz` prefix). No edits to `app_router.py`.

Pydantic models added to `src/core/quiz/quiz_chat/model.py` (kept out of `src/apis/app_model.py` to avoid mixing quiz-chat concerns into the global model file).

## 5. Intent classifier

Two-stage: cheap deterministic rules first, LLM fallback only when rules don't fire.

### 5.1 Stage 1 — Normalisation

```python
def normalise(text: str) -> str:
    # strip → lower → collapse whitespace → strip leading/trailing punctuation
    # keeps internal punctuation so "end-quiz" still tokenises sensibly
```

### 5.2 Stage 2 — Rule-based dispatch

Order of checks (first hit wins):

| Check | Condition | Result |
|---|---|---|
| Letter | `^[abcd]$` (case-insensitive) | `intent=answer`, `answer_index=upper(letter)` |
| Index digit | `^[1-4]$` | `intent=answer`, `answer_index={1:A, 2:B, 3:C, 4:D}` |
| Finish | normalised text in `FINISH_KEYWORDS` (exact) **or** `rapidfuzz.ratio ≥ 85` against any keyword | `intent=finish` |
| Hint | normalised text in `HINT_KEYWORDS` (exact) **or** `rapidfuzz.ratio ≥ 85` | `intent=hint` |

Keyword sets (initial — extensible):

```python
FINISH_KEYWORDS = {
    # English
    "finish", "finish session", "end", "end session", "end quiz",
    "submit", "submit and finish", "done", "quit", "exit", "stop",
    # Vietnamese
    "kết thúc", "ket thuc", "thoát", "thoat", "dừng", "dung",
    "nộp bài", "nop bai",
}

HINT_KEYWORDS = {
    # English
    "hint", "hin", "give me a hint", "help",
    # Vietnamese
    "gợi ý", "goi y",
}
```

Threshold rationale: 85 catches typos like `hin` → `hint` (ratio ~89) and `kết thuc` → `kết thúc` while rejecting unrelated short tokens.

### 5.3 Stage 3 — LLM fallback

Triggered when Stage 2 does not match. Single LLM call. Inputs: `message`, `question`, `options[]`, `language` (detected by re-using the existing `_adetect_language_llm` helper or a simple Unicode check).

LLM output schema:
```json
{
  "intent": "answer" | "question",
  "answer_index": "A" | "B" | "C" | "D" | null,
  "language_match": true | false
}
```

Prompt rules (full text in `prompts.py`):
- Decide whether `message` is intended as an *answer attempt* or an *unrelated question*.
- If answer attempt: pick the option whose meaning is closest to `message`. Return `null` if none is reasonably close.
- `language_match` = whether `message` is in the same primary language as the `question` text. If false → caller will treat as `question`.

### 5.4 Language guard (Scenario 2 warning)

If `language_match` is false, the classifier returns `intent=question` regardless of how confident the answer match was. Reason: AC-1 Scenario 2 explicitly says "Người học phải nhập bằng ngôn ngữ của câu hỏi … Nếu không, hệ thống sẽ xử lý như scenario khác." The "scenario khác" is the free-form question flow.

### 5.5 Ambiguous answer

If `intent=answer` but `answer_index=null` (LLM couldn't pick), the API returns the `answer` shape with a `message` field telling the user to try again. Java backend treats this as a soft error and re-prompts; it does not consume the user's answer attempt.

## 6. Question-intent pipeline

When the classifier returns `intent=question`, the dispatcher delegates to a new function `async_quiz_question_dispatch` in `quiz_chat/pipeline.py`.

### 6.1 Reuse strategy

`async_quiz_question_dispatch` does **not** call `async_pipeline_dispatch` because that function delegates straight into `_arun_core_search` / `_arun_overall_search` which call `_agenerate_answer` with the bare `standalone_query`. We need to inject a no-spoiler instruction at the answer-generation step only — without touching reflection / HyDE / vector search.

Strategy: import the existing private helpers from `src/rag/search/pipeline.py` and assemble a near-identical mini-dispatch in `quiz_chat/pipeline.py`. Steps:

1. `_avalidate_and_prepare(user_input=message, conversation_id=None, llm)` — already skips history when `conversation_id` is None (existing line 142–144). No change needed there.
2. `_aroute_intent(embedder, standalone_query)` — pick core vs. overall.
3. Run the matching `_arun_*_search` *up to* the answer-generation step. Since `_arun_core_search` and `_arun_overall_search` are monolithic, we copy the orchestration into the new pipeline (~30 lines each) and replace the final `_agenerate_answer` call with a quiz-aware variant.

Yes, that is duplication. The alternative — adding an `extra_instruction` parameter to existing pipeline functions — touches files the brief says to leave alone. Duplication contained inside the new module is the lesser evil.

### 6.2 Quiz-aware answer generator

```python
async def _aquiz_generate_answer(
    llm, standalone_query, detected_language, enriched_chunks,
    current_question,
):
    final_prompt = build_final_prompt(
        user_input=standalone_query,
        detected_language=detected_language,
        relevant_chunks=enriched_chunks,
    )
    quiz_guard = QUESTION_GUARD_INSTRUCTION.format(
        node_name=current_question.node_name,
        category=current_question.category,
        question_text=current_question.question,
        options_block=_format_options(current_question.options),
    )
    augmented_prompt = quiz_guard + "\n\n" + final_prompt
    return (await llm.ainvoke_creative(augmented_prompt)).strip()
```

`QUESTION_GUARD_INSTRUCTION` (full text in `prompts.py`):
> The user is currently answering a multiple-choice quiz question on the topic "{node_name}" (category: {category}). The current question is: "{question_text}" with options: {options_block}.
>
> When you respond:
> - Explain the underlying concept clearly using the source material below.
> - Do NOT state which option (A/B/C/D) is the correct answer.
> - Do NOT echo any single option verbatim as the answer.
> - Do NOT phrase your reply as "the answer is X" or equivalent.
> - You may reference the topic and discuss what each related concept means in general; you may quote source material; you may use citations.
>
> Cite sources normally. The user already has the question in front of them — your job is to help them think, not to grade their answer.

### 6.3 Response mapping

`PipelineResult.response` → `content`.
`PipelineResult.sources` → `sources`. The existing `extract_sources` already produces `ChatSourceModel` instances; we serialise via `.model_dump()` like the streaming path does.

Off-topic / no-result / clarity outcomes from the underlying pipeline are returned as `intent=question` with whatever `content` the pipeline produced (clarification text, "no result" message, etc.). Sources are an empty list when `answer_satisfied` is false (matches existing `chat_ask` behaviour at `app_controller.py:80–81`).

## 7. Data flow

```
FE → POST /quiz/chat {message, current_question}
       │
       ▼
quiz_chat_pipeline.async_quiz_chat_dispatch
       │
       ├─ classify_intent(message, options)
       │     ├─ Stage 1: normalise
       │     ├─ Stage 2: rule + rapidfuzz
       │     └─ Stage 3: LLM (only if rules miss)
       │
       ├─ intent == hint   → return {intent: "hint"}
       ├─ intent == finish → return {intent: "finish"}
       ├─ intent == answer → return {intent: "answer", answer_index | null + message}
       └─ intent == question
             ├─ _avalidate_and_prepare(conversation_id=None)
             ├─ _aroute_intent
             ├─ run core/overall search (mirror of existing flow)
             ├─ _aquiz_generate_answer (with QUESTION_GUARD_INSTRUCTION)
             └─ return {intent: "question", content, sources}
```

## 8. Error handling

| Case | Response |
|---|---|
| `message` empty / > 15k chars | FastAPI 422 (Pydantic validation) |
| `current_question` missing required fields | FastAPI 422 |
| LLM classification call fails | Default to `intent=question` and let the question pipeline answer; log exception |
| Question pipeline raises | Wrap in `QdrantApiError` (matches existing `chat_ask` pattern) |
| Underlying pipeline returns unsupported-language / off-topic | Return `intent=question` with the pipeline's templated message and empty sources |

## 9. Testing strategy

Unit tests (`tests/core/quiz/quiz_chat/`):

1. `test_intent_classifier.py`
   - Letter `a/A/b/B/c/C/d/D` → answer with correct index.
   - Digits `1/2/3/4` → answer with correct index.
   - Finish keywords (English + Vietnamese, with and without diacritics) → finish.
   - Hint keywords incl. typo `hin` → hint.
   - Letter with whitespace `" A "` and trailing punctuation `"A."` → answer A.
   - Empty / whitespace-only → falls through to LLM fallback (mock LLM returns `question`).

2. `test_intent_classifier_llm.py` (LLM mocked)
   - Fuzzy text matching option B verbatim → answer B.
   - Fuzzy text near option B but not exact → answer B.
   - Off-topic free-form → question.
   - Vietnamese answer to English question → question (language guard).

3. `test_quiz_chat_pipeline.py` (downstream pipeline mocked)
   - intent=hint/finish: response shape correct, no pipeline call.
   - intent=answer with null index: includes the retry message.
   - intent=question: calls validate_and_prepare with `conversation_id=None`; final prompt sent to LLM contains the guard instruction; response includes content + sources.

4. Manual smoke test against running stack: post sample messages through `/quiz/chat` and verify each intent shape.

## 10. Non-goals

- Streaming response for `intent=question`. The existing `/chat/ask/stream` exists if FE wants streaming later; this endpoint is sync-only for v1.
- Persisting the quiz chat transcript. The new pipeline is fully stateless; persistence is the Java backend's job.
- Enforcing "hint already used" or "must answer before finish" rules. Those are business rules owned by Java.
- Multi-turn quiz reasoning ("explain your previous answer"). No history by design.
