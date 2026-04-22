# Overall Course Topic — Design Spec

**Date:** 2026-04-22
**Author:** danth-ai-swe
**Status:** Approved (ready for implementation plan)

## 1. Background & Goal

The current RAG pipeline routes user queries between two intents: `core_knowledge` (LOMA content lookup against the `imt_kb_v1` Qdrant collection) and `off_topic` (canned redirect). Questions about course-level metadata — "What is LOMA?", "How many modules in LOMA 281?", "What does Module 2 cover?", "Compare LOMA 281 and LOMA 291" — are answered poorly because the source PDFs do not contain that surface-level structural information; they only contain lesson body content.

**Goal:** Add a third intent `overall_course_knowledge` that handles metadata-style questions. Pre-build a small, dense Qdrant collection (`overall_course`) of synthesized chunks derived from `data/metadata_node.json` plus the course/module/lesson taxonomy. Reuse the existing HyDE → hybrid search → LLM filter → answer pipeline 100% — only the target collection differs.

## 2. Scope

In scope:

- New constant `OVERALL_CORE_KNOWLEDGE` and collection name `overall_course`.
- New `courseMetadataSamples` route for the semantic intent router (~30 samples already supplied).
- New script `build_overall_chunks.py` that reads `data/metadata_node.json` and emits 5 chunk types (~380 chunks total — node counts are exact, lesson count derives from data) into `data/ingest_overall/overall_course.json`.
- Refactor `pipeline.py` into a `dispatch` + `_arun_hyde_search` split so the search stage can be invoked against any collection.
- Per-collection `QdrantManager` cache to remove the racy `qdrant.collection_name = X` mutation pattern.
- Wire ingest endpoint to source from `data/ingest_overall/` when `collection_name=overall_course`.

Out of scope:

- Changes to prompt templates (system, clarity, HyDE, filter, summarize) — all reused unchanged.
- Changes to neighbor enrichment, web fallback, source extraction, language detection, or chat history summarization behavior.
- Quiz pipeline.
- Frontend / response model changes.
- Unit test additions (project has no unit test framework yet — verification is manual smoke testing).

## 3. Architecture

### 3.1 Three-intent semantic router

`SemanticRouter` in `src/rag/semantic_router/router.py` already supports N routes. Today only 2 are registered. Add a 3rd:

```python
Route(name=OVERALL_CORE_KNOWLEDGE, samples=courseMetadataSamples)
```

`main.py` is already wired for this (lines 10, 25, 52) — only the constant + sample list need to be added to make imports succeed. The precomputed embedding cache (`cache/embeddings/intent_routes.npz`) auto-invalidates when `ROUTE_SAMPLES` content changes (existing checksum mechanism in `precomputed.py`).

### 3.2 Pipeline split (Approach B)

Current `async_pipeline_hyde_search` is monolithic: validate → quiz check → routing → clarity+HyDE → embed → vector search → enrich → filter → answer. Split into two:

- **`async_pipeline_dispatch(user_input, conversation_id)`** (new public entry point)
  Steps 1–3: validate language + quiz check + intent routing. Picks `collection_name` based on intent, then delegates to `_arun_hyde_search`.

- **`_arun_hyde_search(llm, embedder, collection_name, user_input, standalone_query, detected_language)`** (private)
  Steps 4–9: clarity + HyDE in parallel → embed variants → hybrid vector search → neighbor enrichment → LLM relevance filter → answer generation. Identical to current behavior except that `collection_name` is a parameter.

### 3.3 Per-collection Qdrant client cache

The current pattern mutates `qdrant.collection_name` at the start of each request. With one collection this is benign; with two collections it becomes a real race condition (request A sets "overall", awaits something, request B sets "imt_kb_v1" before A reaches `ahybrid_search`). Fix:

```python
_clients: dict[str, QdrantManager] = {}

def get_qdrant_client(collection_name: str = COLLECTION_NAME) -> QdrantManager:
    if collection_name not in _clients:
        _clients[collection_name] = QdrantManager(collection_name=collection_name)
    return _clients[collection_name]
```

`QdrantManager.__init__` loads a heavy `LateInteractionTextEmbedding` (ColBERT) model. Avoid loading it twice by extracting it into a module-level singleton:

```python
_colbert_singleton: LateInteractionTextEmbedding | None = None

def _get_colbert(model: str = "colbert-ir/colbertv2.0") -> LateInteractionTextEmbedding:
    global _colbert_singleton
    if _colbert_singleton is None:
        _colbert_singleton = LateInteractionTextEmbedding(model)
    return _colbert_singleton
```

`QdrantManager.__init__` calls `_get_colbert(colbert_model)` instead of constructing locally. Same `dense_embedder` already comes from a singleton (`get_openai_embedding_client`).

## 4. Chunk Builder

### 4.1 Source

`data/metadata_node.json` — list of 343 dicts, fields:
`Node ID, Node Name, Definition, Category, Domain Tags, Related Nodes, Source, Summary, Course, Module, Lesson, path`.

Distribution:
- LOMA 281: 230 nodes / 4 modules / 14 lessons
- LOMA 291: 113 nodes / 4 modules / 11 lessons

### 4.2 Five chunk types

| Type | Count | Text body | Payload |
|------|-------|-----------|---------|
| **node** | 343 | `Node {id} — {name}\nDefinition: ...\nCategory: ...\nTags: ...\nRelated Nodes: ...\nCourse: ... \| Module: ... \| Lesson: ...\nPath: ...` | `chunk_type="node"`, `node_id`, `node_name`, `category`, `course`, `module`, `lesson`, `tags`, `file_name` (= `Source` field) |
| **lesson** | ~26 (derived from data) | `Lesson {N} - {title} (Course: {course}, Module: {N})\nNumber of nodes: M\nConcepts covered:\n- {NodeName}: {Definition}\n  ...\nLesson Summary (from source): {Summary of first node in lesson}` | `chunk_type="lesson"`, `course`, `module`, `lesson`, `node_count` |
| **module** | 8 | `Module {N} - {title} (Course: {course})\nNumber of lessons: L\nNumber of nodes: M\nLessons:\n- Lesson 1 - {title} (X nodes)\n  ...\nCategories covered: {comma-separated unique categories, sorted by frequency desc}` | `chunk_type="module"`, `course`, `module`, `lesson_count`, `node_count` |
| **course** | 2 | `{Course full name}\nNumber of modules: 4\nNumber of lessons: L\nNumber of nodes: M\nModules:\n- Module 1 - {title} (X lessons)\n  ...\nMain topics: {top 8 unique categories across the course, sorted by node frequency desc}` | `chunk_type="course"`, `course`, `module_count`, `lesson_count`, `node_count` |
| **overview** | 1 | `LOMA Overview\nLOMA stands for Life Office Management Association.\nLOMA has 2 courses available in this knowledge base:\n- {Course 1 full name}: {N} modules, {N} lessons, {N} nodes\n- {Course 2 full name}: {N} modules, {N} lessons, {N} nodes\n\nFull Course Tree:\n+ {Course 1 full name}\n  + Module 1 - ...\n    - Lesson 1 - ...\n    - Lesson 2 - ...\n    ...\n  + Module 2 - ...\n+ {Course 2 full name}\n  + Module 1 - ...\n    ...` | `chunk_type="overview"` |

**Total: ~380 chunks.** "LOMA stands for Life Office Management Association" is hardcoded in the overview chunk (not derivable from JSON). All other counts and tree structure are computed dynamically from the source JSON at build time.

Lesson chunk uses the `Summary` field from the **first node in that lesson** (per user decision — nodes within a lesson share the same `Summary` field in the source data).

### 4.3 Output

File: `data/ingest_overall/overall_course.json` — list of dicts in the format `aupload_documents` expects:

```json
[
  {
    "id": "<deterministic uuid v5 from chunk_type + identifying keys>",
    "text": "<chunk body>",
    "payload": { "chunk_type": "...", ... }
  }
]
```

Deterministic IDs allow re-running the builder + re-ingesting without producing duplicates.

### 4.4 Entry point

`python -m src.rag.ingest.build_overall_chunks` — reads `METADATA_NODE_JSON`, writes `OVERALL_INGEST_DIR / "overall_course.json"`.

## 5. Ingest Flow

`upload_to_qdrant` in `src/rag/ingest/pipeline.py` gains a `source_dir: Path | None` parameter (defaults to `INGEST_DIR`). Replaces the `manager = get_qdrant_client(); manager.collection_name = X` mutation with `manager = get_qdrant_client(target_collection)`.

`app_controller.py:24` (`/api/v1/documents/ingest`) auto-picks `OVERALL_INGEST_DIR` when `collection_name == OVERALL_COLLECTION_NAME`, otherwise `INGEST_DIR`.

Workflow:

1. `python -m src.rag.ingest.build_overall_chunks` — generate JSON file.
2. `GET /api/v1/documents/ingest?collection_name=overall_course&force_restart=true` — create collection + upload chunks.
3. App restart picks up new intent route automatically (cache invalidates via checksum).
4. `POST /api/v1/chat/ask` with metadata questions → routed to `overall_course_knowledge` → answered from new collection.

## 6. Constants

Add to `src/constants/app_constant.py`:

```python
OVERALL_CORE_KNOWLEDGE: str = "overall_course_knowledge"
OVERALL_COLLECTION_NAME = "overall_course"
OVERALL_INGEST_DIR = DATA_DIR / "ingest_overall"
```

## 7. Reused Components (unchanged behavior)

- All prompt templates: `SYSTEM_PROMPT_TEMPLATE`, `CLARITY_CHECK_PROMPT`, `HYDE_VARIANTS_PROMPT`, `CHUNK_RELEVANCE_FILTER_PROMPT`, `DETECT_LANGUAGE_PROMPT`, `SUMMARIZE_PROMPT_TEMPLATE`.
- Helpers: `_acheck_input_clarity`, `_ahyde_generate_variants`, `_aembed_hyde_variants`, `_avector_search`, `_afetch_neighbor_chunks`, `_merge_chunks`, `_afilter_relevant_chunks`, `_agenerate_answer`, `_avalidate_and_prepare`, `_aroute_intent`, `extract_sources`.
- Web search fallback (`web_rag_answer`) when no chunks / no relevant chunks — same fallback applies to overall intent (rare, since the chunks are dense).
- Neighbor enrichment — synthetic chunks have no `previous`/`next` → existing code already skips via `if prev_list:` checks. No change.
- Source extraction — only node-level chunks populate `file_name` (from JSON `Source` field). Other chunk types have no source → `extract_sources` returns empty for those, which is correct.

## 8. Files Touched

| File | Change |
|------|--------|
| `src/constants/app_constant.py` | Add `OVERALL_CORE_KNOWLEDGE`, `OVERALL_COLLECTION_NAME`, `OVERALL_INGEST_DIR` |
| `src/rag/semantic_router/samples.py` | Add `courseMetadataSamples` |
| `src/rag/semantic_router/precomputed.py` | Add overall route to `ROUTE_SAMPLES` |
| `src/rag/db_vector.py` | Add `_get_colbert()` singleton; refactor `get_qdrant_client(collection_name)` to per-collection cache |
| `src/rag/ingest/pipeline.py` | Add `source_dir` param; remove `collection_name` mutation pattern |
| `src/rag/ingest/build_overall_chunks.py` | **NEW** — chunk builder script |
| `src/rag/search/pipeline.py` | Split into `async_pipeline_dispatch` (new public) + `_arun_hyde_search` (private). Remove `qdrant.collection_name = X` mutation |
| `src/apis/app_controller.py` | Switch entry from `async_pipeline_hyde_search` → `async_pipeline_dispatch`; add `source_dir` selection in `/ingest` |

## 9. Verification (manual smoke test)

1. Run `python -m src.rag.ingest.build_overall_chunks` — confirm output file has expected count (~380 chunks) and well-formed text bodies.
2. Boot app — confirm intent router cache rebuilds with 3 routes (logs).
3. Trigger `/ingest?collection_name=overall_course` — confirm collection appears in Qdrant with ~380 points.
4. Smoke-test queries from `courseMetadataSamples`:
   - "What is LOMA?" → overview chunk hit, mentions "Life Office Management Association".
   - "How many modules in LOMA 281?" → course chunk hit, answers "4".
   - "What is covered in Module 2 of LOMA 281?" → module chunk hit, lists 4 lessons.
   - "Compare LOMA 281 and LOMA 291." → both course chunks + overview hit.
   - "What is the definition of underwriting in LOMA 281?" → node chunk hit.
5. Verify a core knowledge query (e.g. "Explain adverse selection in detail") still routes to `core_knowledge` and queries `imt_kb_v1` correctly.
6. Verify off-topic query still returns canned message.

## 10. Risks

- **Race condition fix is touching shared infrastructure.** `db_vector.py` change affects all callers of `get_qdrant_client`. Mitigation: signature stays backward compatible (default `COLLECTION_NAME`), and the only mutation site (`ingest/pipeline.py`) is updated in the same change.
- **Intent router accuracy.** Adding a 3rd route may cause borderline queries (e.g., "What is risk in LOMA 281?") to misroute between `core_knowledge` and `overall_course_knowledge`. Sample sets must be tuned if smoke testing reveals confusion. Same `top_k_samples=15` mean-cosine logic applies.
- **Web fallback for overall intent.** Per user decision, kept identical to core pipeline. Should rarely fire because chunks are dense; if it does fire on metadata questions, the web answer may be misleading. Acceptable tradeoff for now.
- **No automated tests** — verification is manual. Future work could add a regression set of routing samples + golden answers.
