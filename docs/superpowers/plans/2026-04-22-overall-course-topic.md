# Overall Course Topic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third intent `overall_course_knowledge` that answers course-metadata questions (e.g. "How many modules in LOMA 281?", "What is LOMA?") by routing to a pre-built `overall_course` Qdrant collection while reusing the existing HyDE → hybrid search → LLM filter → answer pipeline 100%.

**Architecture:** Split `pipeline.py` into a `dispatch` (validate + quiz + intent routing) and `_arun_hyde_search` (collection-scoped search). Add per-collection cached `QdrantManager` to remove the racy `qdrant.collection_name = X` mutation. Build a chunk-builder script that synthesizes 5 chunk types (~380 chunks total) from `data/metadata_node.json` plus the LOMA taxonomy.

**Tech Stack:** Python 3.11+, FastAPI, Qdrant (hybrid dense + sparse + ColBERT), Azure OpenAI embeddings/chat, fastembed.

**Spec:** `docs/superpowers/specs/2026-04-22-overall-course-topic-design.md`

**Test approach:** Project has no pytest framework. Verification is done via `python -c "..."` import/signature checks, REPL inspection of script output, and manual smoke tests against a running app + Qdrant. Each task includes a verification step before commit.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/constants/app_constant.py` | Modify | Add new constants for overall topic |
| `src/rag/semantic_router/samples.py` | Modify | Add `courseMetadataSamples` route samples |
| `src/rag/semantic_router/precomputed.py` | Modify | Register overall route in `ROUTE_SAMPLES` |
| `src/rag/db_vector.py` | Modify | ColBERT singleton + per-collection `QdrantManager` cache |
| `src/rag/ingest/pipeline.py` | Modify | Accept `source_dir` arg; remove `collection_name` mutation |
| `src/rag/ingest/build_overall_chunks.py` | Create | Chunk builder script (5 chunk types) |
| `src/rag/search/pipeline.py` | Modify | Split `async_pipeline_dispatch` (new public) + `_arun_hyde_search` (private) |
| `src/apis/app_controller.py` | Modify | Use dispatch entry point; auto-pick `source_dir` for overall ingest |

---

## Task 1: Add new constants

**Files:**
- Modify: `src/constants/app_constant.py`

- [ ] **Step 1: Add overall constants right after the existing intent constants**

Open `src/constants/app_constant.py`. Find the block (lines 44-46):

```python
INTENT_CORE_KNOWLEDGE: str = "core_knowledge"
INTENT_OFF_TOPIC: str = "off_topic"
INTENT_QUIZ: str = "quiz"
```

Replace with:

```python
INTENT_CORE_KNOWLEDGE: str = "core_knowledge"
INTENT_OFF_TOPIC: str = "off_topic"
INTENT_QUIZ: str = "quiz"
INTENT_OVERALL_COURSE_KNOWLEDGE: str = "overall_course_knowledge"

OVERALL_COLLECTION_NAME = "overall_course"
OVERALL_INGEST_DIR = DATA_DIR / "ingest_overall"
```

- [ ] **Step 2: Verify imports succeed**

Run from project root:

```bash
python -c "from src.constants.app_constant import INTENT_OVERALL_COURSE_KNOWLEDGE, OVERALL_COLLECTION_NAME, OVERALL_INGEST_DIR; print(INTENT_OVERALL_COURSE_KNOWLEDGE, OVERALL_COLLECTION_NAME, OVERALL_INGEST_DIR)"
```

Expected output:
```
overall_course_knowledge overall_course D:\Deverlopment\huudan.com\PythonProject\data\ingest_overall
```

- [ ] **Step 3: Commit**

```bash
git add src/constants/app_constant.py
git commit -m "feat: add OVERALL_* constants for overall course topic"
```

---

## Task 2: Add `courseMetadataSamples` to samples module

**Files:**
- Modify: `src/rag/semantic_router/samples.py`

- [ ] **Step 1: Append `courseMetadataSamples` to `samples.py`**

Open `src/rag/semantic_router/samples.py`. After the existing `offTopicSamples` list (after line 30), append:

```python

courseMetadataSamples = [
    # Về khóa học tổng quát
    "What is LOMA?",
    "What does LOMA stand for?",
    "What is LOMA 281?",
    "What is LOMA 291?",
    "Give me an overview of LOMA 281.",
    "Give me an overview of LOMA 291.",
    "What is the difference between LOMA 281 and LOMA 291?",
    "Compare LOMA 281 and LOMA 291.",

    # Về cấu trúc khóa học
    "How many modules are in LOMA 281?",
    "How many lessons does LOMA 291 have?",
    "What modules are covered in LOMA 281?",
    "List all lessons in LOMA 291.",
    "What is covered in Module 2 of LOMA 281?",
    "What is the structure of LOMA 291?",

    # Định nghĩa / tóm tắt node
    "What is the definition of underwriting in LOMA 281?",
    "Summarize lesson 3 of LOMA 281.",
    "What does Module 1 of LOMA 291 cover?",
    "Give me a summary of the lesson on risk management.",
    "What topics are tagged under 'reinsurance' in LOMA?",
    "Which lessons belong to the 'Underwriting' domain?",
    "What category does the lesson on premiums fall under?",

    # So sánh khái niệm cơ bản
    "Compare the definitions of risk and uncertainty as taught in LOMA.",
    "What is the difference between a module and a lesson in LOMA?",
    "How are topics organized in LOMA 281 vs LOMA 291?",

    # Tra cứu node metadata
    "What are the tags for the lesson on beneficiaries?",
    "What domain does LOMA 281 Module 3 belong to?",
    "Find all lessons related to 'mortality risk'.",
    "What node covers the concept of insurable interest?",
]
```

- [ ] **Step 2: Verify import + count**

Run:

```bash
python -c "from src.rag.semantic_router.samples import courseMetadataSamples; print(f'count={len(courseMetadataSamples)}'); print(courseMetadataSamples[0])"
```

Expected output:
```
count=28
What is LOMA?
```

- [ ] **Step 3: Verify `main.py` import line 25 now succeeds**

Run:

```bash
python -c "from src.rag.semantic_router.samples import offTopicSamples, coreKnowledgeSamples, courseMetadataSamples; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/rag/semantic_router/samples.py
git commit -m "feat: add courseMetadataSamples for overall course route"
```

---

## Task 3: Register overall route in precomputed embeddings

**Files:**
- Modify: `src/rag/semantic_router/precomputed.py`

- [ ] **Step 1: Update imports + add overall route to `ROUTE_SAMPLES`**

Open `src/rag/semantic_router/precomputed.py`. Replace lines 7-19 (imports + `ROUTE_SAMPLES` definition):

```python
from src.rag.search.pipeline import INTENT_CORE_KNOWLEDGE, INTENT_OFF_TOPIC
from src.rag.semantic_router.samples import offTopicSamples, coreKnowledgeSamples
from src.utils.logger_utils import logger

CACHE_DIR = Path("cache/embeddings")
CACHE_FILE = CACHE_DIR / "intent_routes.npz"
CHECKSUM_FILE = CACHE_DIR / "intent_routes.checksum"

ROUTE_SAMPLES: Dict[str, list] = {
    INTENT_CORE_KNOWLEDGE: coreKnowledgeSamples,
    INTENT_OFF_TOPIC: offTopicSamples,
}
```

with:

```python
from src.constants.app_constant import (
    INTENT_CORE_KNOWLEDGE, INTENT_OFF_TOPIC, INTENT_OVERALL_COURSE_KNOWLEDGE,
)
from src.rag.semantic_router.samples import (
    offTopicSamples, coreKnowledgeSamples, courseMetadataSamples,
)
from src.utils.logger_utils import logger

CACHE_DIR = Path("cache/embeddings")
CACHE_FILE = CACHE_DIR / "intent_routes.npz"
CHECKSUM_FILE = CACHE_DIR / "intent_routes.checksum"

ROUTE_SAMPLES: Dict[str, list] = {
    INTENT_CORE_KNOWLEDGE: coreKnowledgeSamples,
    INTENT_OFF_TOPIC: offTopicSamples,
    INTENT_OVERALL_COURSE_KNOWLEDGE: courseMetadataSamples,
}
```

(Note: import source for `INTENT_*` switches from `pipeline.py` → `app_constant.py`. This breaks a circular dep risk and is the cleaner source.)

- [ ] **Step 2: Verify import + cache invalidation logic**

Run:

```bash
python -c "from src.rag.semantic_router.precomputed import ROUTE_SAMPLES, _compute_checksum, _is_cache_valid; print('routes:', list(ROUTE_SAMPLES.keys())); print('cache_valid:', _is_cache_valid())"
```

Expected output:
```
routes: ['core_knowledge', 'off_topic', 'overall_course_knowledge']
cache_valid: False
```

(`cache_valid: False` is correct — adding a new route changes the checksum so stale cache is invalidated. The app will rebuild embeddings on next boot.)

- [ ] **Step 3: Commit**

```bash
git add src/rag/semantic_router/precomputed.py
git commit -m "feat: register overall_course_knowledge route in semantic router cache"
```

---

## Task 4: Refactor `db_vector.py` — ColBERT singleton + per-collection client cache

**Files:**
- Modify: `src/rag/db_vector.py`

This task fixes a pre-existing race condition that becomes load-bearing once we have multiple collections.

- [ ] **Step 1: Add ColBERT module-level singleton**

Open `src/rag/db_vector.py`. After the existing imports + `config = get_app_config()` line (around line 19), add:

```python
_colbert_singleton: LateInteractionTextEmbedding | None = None


def _get_colbert(model: str = "colbert-ir/colbertv2.0") -> LateInteractionTextEmbedding:
    """ColBERT model is heavy (~hundreds MB) — cache at module level so every
    QdrantManager instance shares the same loaded model."""
    global _colbert_singleton
    if _colbert_singleton is None:
        _colbert_singleton = LateInteractionTextEmbedding(model)
    return _colbert_singleton
```

- [ ] **Step 2: Use ColBERT singleton in `QdrantManager.__init__`**

Find lines 22-34 (the `QdrantManager.__init__` method). Replace:

```python
    def __init__(
            self,
            collection_name: str,
            url: str = config.QDRANT_URL,
            colbert_model: str = "colbert-ir/colbertv2.0",
    ) -> None:
        self.collection_name = collection_name
        api_key = config.QDRANT_APIKEY if config.PROFILE_NAME == "prod" else None
        self.client = QdrantClient(url=url, api_key=api_key)
        self.async_client = AsyncQdrantClient(url=url, api_key=api_key)
        self.dense_embedder = get_openai_embedding_client()
        self._colbert = LateInteractionTextEmbedding(colbert_model)
```

with:

```python
    def __init__(
            self,
            collection_name: str,
            url: str = config.QDRANT_URL,
            colbert_model: str = "colbert-ir/colbertv2.0",
    ) -> None:
        self.collection_name = collection_name
        api_key = config.QDRANT_APIKEY if config.PROFILE_NAME == "prod" else None
        self.client = QdrantClient(url=url, api_key=api_key)
        self.async_client = AsyncQdrantClient(url=url, api_key=api_key)
        self.dense_embedder = get_openai_embedding_client()
        self._colbert = _get_colbert(colbert_model)
```

- [ ] **Step 3: Replace bottom-of-file singleton with per-collection cache**

Find lines 309-316 (bottom of file):

```python
_client_instance: "QdrantManager | None" = None


def get_qdrant_client() -> QdrantManager:
    global _client_instance
    if _client_instance is None:
        _client_instance = QdrantManager(collection_name=COLLECTION_NAME)
    return _client_instance
```

Replace with:

```python
_clients: dict[str, QdrantManager] = {}


def get_qdrant_client(collection_name: str = COLLECTION_NAME) -> QdrantManager:
    """Per-collection cached QdrantManager. Avoids the race condition that
    arises when multiple requests mutate `manager.collection_name` to point
    at different collections simultaneously."""
    if collection_name not in _clients:
        _clients[collection_name] = QdrantManager(collection_name=collection_name)
    return _clients[collection_name]
```

- [ ] **Step 4: Verify per-collection caching**

Run:

```bash
python -c "from src.rag.db_vector import get_qdrant_client, _get_colbert; a = get_qdrant_client('foo'); b = get_qdrant_client('bar'); c = get_qdrant_client('foo'); print('a is c:', a is c); print('a is b:', a is b); print('a.collection:', a.collection_name); print('b.collection:', b.collection_name); print('shared colbert:', a._colbert is b._colbert)"
```

Expected output:
```
a is c: True
a is b: False
a.collection: foo
b.collection: bar
shared colbert: True
```

- [ ] **Step 5: Commit**

```bash
git add src/rag/db_vector.py
git commit -m "refactor: per-collection QdrantManager cache + ColBERT singleton"
```

---

## Task 5: Refactor `ingest/pipeline.py` — accept `source_dir`, remove mutation

**Files:**
- Modify: `src/rag/ingest/pipeline.py`

- [ ] **Step 1: Update imports**

Open `src/rag/ingest/pipeline.py`. Replace the existing import block (lines 1-15):

```python
import json

from qdrant_client import models

from src.constants.app_constant import COLLECTION_NAME
from src.constants.app_constant import (
    INGEST_DIR,
)
from src.rag.db_vector import get_qdrant_client
from src.utils.checkpoint_utils import (
    load_checkpoint as _load_checkpoint_raw,
    save_checkpoint as _save_checkpoint_raw,
    clear_checkpoint,
)
from src.utils.logger_utils import logger, StepTimer
```

with:

```python
import json
from pathlib import Path

from qdrant_client import models

from src.constants.app_constant import COLLECTION_NAME, INGEST_DIR
from src.rag.db_vector import get_qdrant_client
from src.utils.checkpoint_utils import (
    load_checkpoint as _load_checkpoint_raw,
    save_checkpoint as _save_checkpoint_raw,
    clear_checkpoint,
)
from src.utils.logger_utils import logger, StepTimer
```

- [ ] **Step 2: Update `upload_to_qdrant` signature + body**

Find the function `upload_to_qdrant` (starts line 29). Replace the entire function body with:

```python
async def upload_to_qdrant(
        force_restart: bool = False,
        collection_name: str | None = None,
        source_dir: Path | None = None,
):
    target_collection = collection_name or COLLECTION_NAME
    target_dir = source_dir or INGEST_DIR
    timer = StepTimer("ingest_upload")

    try:
        async with timer.astep("prepare_checkpoint"):
            if force_restart:
                clear_checkpoint()

            uploaded_files = load_checkpoint()
            all_json_files = sorted(target_dir.glob("*.json"))
            pending_files = [f for f in all_json_files if f.name not in uploaded_files]

        logger.info(f"📊 Tổng file: {len(all_json_files)}")
        logger.info(f"✅ Đã upload:  {len(uploaded_files)}")
        logger.info(f"⏳ Còn lại:    {len(pending_files)}\n")

        if not pending_files:
            logger.info("🎉 Tất cả file đã được upload rồi!")
            return

        async with timer.astep("create_collection_and_indexes"):
            manager = get_qdrant_client(target_collection)
            await manager.acreate_collection(recreate=force_restart)
            await manager.acreate_payload_index("category", models.PayloadSchemaType.KEYWORD)
            await manager.acreate_payload_index("node_name", models.PayloadSchemaType.KEYWORD)
            await manager.acreate_payload_index("course", models.PayloadSchemaType.KEYWORD)
            await manager.acreate_payload_index("module", models.PayloadSchemaType.INTEGER)
            await manager.acreate_payload_index("lesson", models.PayloadSchemaType.INTEGER)

        for i, json_file in enumerate(pending_files, 1):
            logger.info(f"[{i}/{len(pending_files)}] 📂 {json_file.name} ...")

            async with timer.astep(f"upload_file_{json_file.stem}"):
                with open(json_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Lỗi parse JSON: {e}")
                        continue

                if not isinstance(data, list):
                    logger.error(f"❌ Không phải list, bỏ qua.")
                    continue

                try:
                    await manager.aupload_documents(
                        data,
                        batch_size=50,
                        max_retries=3,
                    )
                    uploaded_files.add(json_file.name)
                    save_checkpoint(uploaded_files)
                    logger.info(f"✅ {len(data)} chunks")

                except Exception as e:
                    logger.exception(f"❌ Upload thất bại: {json_file.name}")
                    logger.info("💾 Checkpoint đã lưu tiến độ, chạy lại để tiếp tục.")
                    return

        logger.info(f"\n🚀 Hoàn tất! Đã upload {len(uploaded_files)}/{len(all_json_files)} files.")
    finally:
        timer.summary()
```

Key changes vs. original:
- Added `source_dir: Path | None = None` parameter.
- Added `target_dir = source_dir or INGEST_DIR` and use it instead of `INGEST_DIR` for globbing.
- Replaced `manager = get_qdrant_client(); manager.collection_name = target_collection` with `manager = get_qdrant_client(target_collection)`.

- [ ] **Step 3: Verify signature**

Run:

```bash
python -c "import inspect; from src.rag.ingest.pipeline import upload_to_qdrant; sig = inspect.signature(upload_to_qdrant); print(sig)"
```

Expected output:
```
(force_restart: bool = False, collection_name: str | None = None, source_dir: pathlib.Path | None = None)
```

- [ ] **Step 4: Commit**

```bash
git add src/rag/ingest/pipeline.py
git commit -m "refactor: ingest pipeline accepts source_dir, drops collection_name mutation"
```

---

## Task 6: Build chunk-builder script

**Files:**
- Create: `src/rag/ingest/build_overall_chunks.py`

- [ ] **Step 1: Create the script with full implementation**

Create `src/rag/ingest/build_overall_chunks.py` with this content:

```python
"""
Build text chunks tổng hợp cho intent overall_course_knowledge.
Đọc data/metadata_node.json → tạo 5 loại chunk (node/lesson/module/course/overview)
→ ghi ra data/ingest_overall/overall_course.json.

Chạy: python -m src.rag.ingest.build_overall_chunks
"""
import json
import re
import uuid
from collections import Counter, defaultdict
from typing import Any

from src.constants.app_constant import METADATA_NODE_JSON, OVERALL_INGEST_DIR

# Namespace cố định để generate deterministic UUID v5 — cho phép
# rebuild + re-ingest mà không tạo duplicate point.
_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

_TOP_CATEGORIES_COURSE = 8


def _det_id(*parts: str) -> str:
    name = "::".join(parts)
    return str(uuid.uuid5(_NAMESPACE, name))


def _module_number(module_str: str) -> int:
    m = re.match(r"Module\s+(\d+)", module_str)
    return int(m.group(1)) if m else 0


def _lesson_number(lesson_str: str) -> int:
    m = re.match(r"Lesson\s+(\d+)", lesson_str)
    return int(m.group(1)) if m else 0


def _build_node_chunks(rows: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for r in rows:
        text_parts = [
            f"Node {r['Node ID']} — {r['Node Name']}",
            f"Definition: {r['Definition']}",
            f"Category: {r['Category']}",
            f"Tags: {r['Domain Tags']}",
            f"Related Nodes: {r['Related Nodes']}",
            f"Course: {r['Course']} | Module: {r['Module']} | Lesson: {r['Lesson']}",
            f"Path: {r['path']}",
        ]
        chunks.append({
            "id": _det_id("node", str(r["Node ID"])),
            "text": "\n".join(text_parts),
            "payload": {
                "chunk_type": "node",
                "node_id": int(r["Node ID"]),
                "node_name": r["Node Name"],
                "category": r["Category"],
                "course": r["Course"],
                "module": _module_number(r["Module"]),
                "lesson": _lesson_number(r["Lesson"]),
                "tags": r["Domain Tags"],
                "file_name": r["Source"],
            },
        })
    return chunks


def _build_lesson_chunks(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["Course"], r["Module"], r["Lesson"])
        grouped[key].append(r)

    chunks: list[dict] = []
    for (course, module, lesson), nodes in grouped.items():
        first = nodes[0]
        concept_lines = [f"- {n['Node Name']}: {n['Definition']}" for n in nodes]
        text = "\n".join([
            f"{lesson} (Course: {course}, {module})",
            f"Number of nodes: {len(nodes)}",
            "Concepts covered:",
            *concept_lines,
            "",
            f"Lesson Summary (from source): {first['Summary']}",
        ])
        chunks.append({
            "id": _det_id("lesson", course, module, lesson),
            "text": text,
            "payload": {
                "chunk_type": "lesson",
                "course": course,
                "module": _module_number(module),
                "lesson": _lesson_number(lesson),
                "node_count": len(nodes),
            },
        })
    return chunks


def _build_module_chunks(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["Course"], r["Module"])].append(r)

    chunks: list[dict] = []
    for (course, module), nodes in grouped.items():
        lesson_nodes: dict[str, list[dict]] = defaultdict(list)
        for n in nodes:
            lesson_nodes[n["Lesson"]].append(n)
        sorted_lessons = sorted(
            lesson_nodes.items(), key=lambda x: _lesson_number(x[0])
        )
        lesson_lines = [
            f"- {lesson} ({len(nlist)} nodes)"
            for lesson, nlist in sorted_lessons
        ]
        cat_counts = Counter(n["Category"] for n in nodes)
        cats_sorted = [c for c, _ in cat_counts.most_common()]

        text = "\n".join([
            f"{module} (Course: {course})",
            f"Number of lessons: {len(sorted_lessons)}",
            f"Number of nodes: {len(nodes)}",
            "Lessons:",
            *lesson_lines,
            "",
            f"Categories covered: {', '.join(cats_sorted)}",
        ])
        chunks.append({
            "id": _det_id("module", course, module),
            "text": text,
            "payload": {
                "chunk_type": "module",
                "course": course,
                "module": _module_number(module),
                "lesson_count": len(sorted_lessons),
                "node_count": len(nodes),
            },
        })
    return chunks


def _build_course_chunks(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["Course"]].append(r)

    chunks: list[dict] = []
    for course, nodes in grouped.items():
        module_nodes: dict[str, list[dict]] = defaultdict(list)
        lesson_set: set[tuple[str, str]] = set()
        for n in nodes:
            module_nodes[n["Module"]].append(n)
            lesson_set.add((n["Module"], n["Lesson"]))

        sorted_modules = sorted(
            module_nodes.items(), key=lambda x: _module_number(x[0])
        )
        module_lines: list[str] = []
        for module, mnodes in sorted_modules:
            mlessons = {n["Lesson"] for n in mnodes}
            module_lines.append(f"- {module} ({len(mlessons)} lessons)")

        cat_counts = Counter(n["Category"] for n in nodes)
        top_cats = [c for c, _ in cat_counts.most_common(_TOP_CATEGORIES_COURSE)]

        text = "\n".join([
            course,
            f"Number of modules: {len(sorted_modules)}",
            f"Number of lessons: {len(lesson_set)}",
            f"Number of nodes: {len(nodes)}",
            "Modules:",
            *module_lines,
            "",
            f"Main topics: {', '.join(top_cats)}",
        ])
        chunks.append({
            "id": _det_id("course", course),
            "text": text,
            "payload": {
                "chunk_type": "course",
                "course": course,
                "module_count": len(sorted_modules),
                "lesson_count": len(lesson_set),
                "node_count": len(nodes),
            },
        })
    return chunks


def _build_overview_chunk(rows: list[dict]) -> dict:
    tree: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    course_node_count: Counter = Counter()
    course_lesson_set: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for r in rows:
        course = r["Course"]
        module = r["Module"]
        lesson = r["Lesson"]
        tree[course][module].add(lesson)
        course_node_count[course] += 1
        course_lesson_set[course].add((module, lesson))

    courses_sorted = sorted(tree.keys())

    lines = [
        "LOMA Overview",
        "LOMA stands for Life Office Management Association.",
        f"LOMA has {len(courses_sorted)} courses available in this knowledge base:",
    ]
    for course in courses_sorted:
        modules_count = len(tree[course])
        lessons_count = len(course_lesson_set[course])
        nodes_count = course_node_count[course]
        lines.append(
            f"- {course}: {modules_count} modules, {lessons_count} lessons, {nodes_count} nodes"
        )

    lines.append("")
    lines.append("Full Course Tree:")
    for course in courses_sorted:
        lines.append(f"+ {course}")
        modules_sorted = sorted(tree[course].keys(), key=_module_number)
        for module in modules_sorted:
            lines.append(f"  + {module}")
            lessons_sorted = sorted(tree[course][module], key=_lesson_number)
            for lesson in lessons_sorted:
                lines.append(f"    - {lesson}")

    return {
        "id": _det_id("overview"),
        "text": "\n".join(lines),
        "payload": {"chunk_type": "overview"},
    }


def build_all_chunks(rows: list[dict]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    chunks.extend(_build_node_chunks(rows))
    chunks.extend(_build_lesson_chunks(rows))
    chunks.extend(_build_module_chunks(rows))
    chunks.extend(_build_course_chunks(rows))
    chunks.append(_build_overview_chunk(rows))
    return chunks


def main() -> None:
    with open(METADATA_NODE_JSON, "r", encoding="utf-8") as f:
        rows = json.load(f)

    chunks = build_all_chunks(rows)

    OVERALL_INGEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OVERALL_INGEST_DIR / "overall_course.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    counts = Counter(c["payload"]["chunk_type"] for c in chunks)
    print(f"✅ Wrote {len(chunks)} chunks → {out_path}")
    for ctype, cnt in sorted(counts.items()):
        print(f"   {ctype}: {cnt}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
python -m src.rag.ingest.build_overall_chunks
```

Expected output (counts may differ slightly based on the exact contents of `metadata_node.json`):
```
✅ Wrote 380 chunks → D:\Deverlopment\huudan.com\PythonProject\data\ingest_overall\overall_course.json
   course: 2
   lesson: 26
   module: 8
   node: 343
   overview: 1
```

(`node: 343` and `module: 8` are exact — verified from current data. `lesson` ≈ 25-26 depending on whether all lessons in the user's tree have at least one node row. `course: 2` and `overview: 1` are exact.)

- [ ] **Step 3: Inspect a sample of each chunk type**

Run:

```bash
python -c "import json; data = json.load(open('data/ingest_overall/overall_course.json', 'r', encoding='utf-8')); types = {}; [types.setdefault(c['payload']['chunk_type'], c) for c in data]; [print('===', t, '==='); print(c['text'][:400]) or print() for t, c in types.items()]"
```

Expected: prints a sample of each of the 5 chunk types with sensible text bodies.

Manually verify:
- The `node` sample includes Definition + Category + Path.
- The `lesson` sample lists "Concepts covered" with multiple `- NodeName: Definition` lines + a "Lesson Summary (from source):" trailer.
- The `module` sample lists 3-4 lessons + "Categories covered: ...".
- The `course` sample lists 4 modules + "Main topics: ...".
- The `overview` sample mentions "Life Office Management Association" + lists 2 courses + has a "Full Course Tree:" section.

- [ ] **Step 4: Commit**

```bash
git add src/rag/ingest/build_overall_chunks.py data/ingest_overall/overall_course.json
git commit -m "feat: add overall course chunk builder + initial generated output"
```

(Committing the generated JSON makes ingest reproducible without re-running the builder.)

---

## Task 7: Refactor `pipeline.py` — split dispatch + collection-scoped search

**Files:**
- Modify: `src/rag/search/pipeline.py`

- [ ] **Step 1: Update imports**

Open `src/rag/search/pipeline.py`. Replace the imports block (lines 10-15):

```python
from src.constants.app_constant import (
    COLLECTION_NAME,
    MAX_INPUT_CHARS,
    INTENT_CORE_KNOWLEDGE, INTENT_OFF_TOPIC, INTENT_QUIZ, NEIGHBOR_PREV_INDEX, VECTOR_SEARCH_TOP_K,
    NEIGHBOR_NEXT_INDEX, UNSUPPORTED_LANGUAGE_MSG, INPUT_TOO_LONG_RESPONSE, OFF_TOPIC_RESPONSE_MAP,
)
```

with:

```python
from src.constants.app_constant import (
    COLLECTION_NAME,
    MAX_INPUT_CHARS,
    INTENT_CORE_KNOWLEDGE, INTENT_OFF_TOPIC, INTENT_QUIZ, NEIGHBOR_PREV_INDEX, VECTOR_SEARCH_TOP_K,
    NEIGHBOR_NEXT_INDEX, UNSUPPORTED_LANGUAGE_MSG, INPUT_TOO_LONG_RESPONSE, OFF_TOPIC_RESPONSE_MAP,
    INTENT_OVERALL_COURSE_KNOWLEDGE, OVERALL_COLLECTION_NAME,
)
```

- [ ] **Step 2: Replace `async_pipeline_hyde_search` with split functions**

Find the entire function `async_pipeline_hyde_search` (starts line 395, ends line 481). Replace it with two functions:

```python
async def _arun_hyde_search(
        llm: AzureChatClient,
        embedder: AzureEmbeddingClient,
        collection_name: str,
        user_input: str,
        standalone_query: str,
        detected_language: str,
) -> PipelineResult:
    timer = StepTimer(f"hyde_search:{collection_name}")
    config: AppConfig = get_app_config()

    try:
        qdrant = get_qdrant_client(collection_name)

        # ── 5a+5b. clarity_check VÀ hyde_variants chạy song song ─────────────
        async with timer.astep("clarity_and_hyde_parallel"):
            clarity_task = _acheck_input_clarity(llm, standalone_query, detected_language)
            hyde_task = _ahyde_generate_variants(llm, standalone_query, detected_language)
            clarity, variants = await asyncio.gather(clarity_task, hyde_task)

        if not clarity.get("clear", True):
            logger.info(f"Input unclear (type={clarity.get('type')}) — returning clarification")
            return _make_clarity_result(clarity, detected_language)

        async with timer.astep("hyde_embedding"):
            mean_dense, mean_colbert = await _aembed_hyde_variants(
                embedder, qdrant, variants
            )

        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, mean_dense, mean_colbert, standalone_query
            )

        if not sorted_chunks:
            logger.info("No vector results — triggering web search fallback")
            async with timer.astep("web_search_fallback_no_chunks"):
                rs = await web_rag_answer(llm, embedder, standalone_query, detected_language)
            return _make_web_search_result(rs["answer"], rs["sources"], detected_language)

        async with timer.astep("enrich_chunks"):
            neighbor_chunks = await _afetch_neighbor_chunks(qdrant, sorted_chunks)
            enriched_chunks = _merge_chunks(sorted_chunks, neighbor_chunks)

        async with timer.astep("filter_relevant_chunks"):
            filtered_chunks = await _afilter_relevant_chunks(
                llm, standalone_query, enriched_chunks
            )

        if not filtered_chunks:
            logger.info("No relevant chunks after filtering — triggering web search fallback")
            async with timer.astep("web_search_fallback_no_relevant"):
                rs = await web_rag_answer(llm, embedder, standalone_query, detected_language)
            return _make_web_search_result(rs["answer"], rs["sources"], detected_language)

        async with timer.astep("generate_answer"):
            answer = await _agenerate_answer(
                llm, standalone_query, detected_language,
                filtered_chunks,
            )

        return _make_rag_result(
            response=answer,
            sources=extract_sources(sorted_chunks, config.APP_DOMAIN),
            detected_language=detected_language,
            answer_satisfied=True,
        )

    finally:
        timer.summary()


async def async_pipeline_dispatch(
        user_input: str,
        conversation_id: Optional[str] = None,
) -> PipelineResult:
    timer = StepTimer("dispatch")

    try:
        llm = get_openai_chat_client()
        embedder = get_openai_embedding_client()

        async with timer.astep("validate_and_prepare"):
            detected_language, standalone_query, error_response = (
                await _avalidate_and_prepare(user_input, conversation_id, llm)
            )
            if error_response:
                return _make_off_topic_result(error_response, "English")

        async with timer.astep("quiz_keyword_check"):
            if is_quiz_intent(user_input) or is_quiz_intent(standalone_query):
                return _make_quiz_result()

        async with timer.astep("intent_routing"):
            intent = await _aroute_intent(embedder, standalone_query)

        if intent == INTENT_OFF_TOPIC:
            return _make_off_topic_result(
                OFF_TOPIC_RESPONSE_MAP.get(detected_language), detected_language
            )

        # Pick collection theo intent. Mọi intent ngoài OVERALL → core collection.
        collection_name = (
            OVERALL_COLLECTION_NAME if intent == INTENT_OVERALL_COURSE_KNOWLEDGE
            else COLLECTION_NAME
        )

        return await _arun_hyde_search(
            llm=llm,
            embedder=embedder,
            collection_name=collection_name,
            user_input=user_input,
            standalone_query=standalone_query,
            detected_language=detected_language,
        )

    finally:
        timer.summary()
```

Note: `async_pipeline_hyde_search` is removed entirely. The only public entry is now `async_pipeline_dispatch`. `_arun_hyde_search` is module-private (underscore prefix).

- [ ] **Step 3: Verify imports + functions exist**

Run:

```bash
python -c "from src.rag.search.pipeline import async_pipeline_dispatch, _arun_hyde_search; import inspect; print('dispatch:', inspect.signature(async_pipeline_dispatch)); print('search:', inspect.signature(_arun_hyde_search))"
```

Expected output:
```
dispatch: (user_input: str, conversation_id: Optional[str] = None) -> src.rag.search.model.PipelineResult
search: (llm: src.rag.llm.chat_llm.AzureChatClient, embedder: src.rag.llm.embedding_llm.AzureEmbeddingClient, collection_name: str, user_input: str, standalone_query: str, detected_language: str) -> src.rag.search.model.PipelineResult
```

- [ ] **Step 4: Verify old name is gone**

Run:

```bash
python -c "from src.rag.search import pipeline; print('async_pipeline_hyde_search' in dir(pipeline))"
```

Expected output: `False`

- [ ] **Step 5: Commit**

```bash
git add src/rag/search/pipeline.py
git commit -m "refactor: split pipeline into dispatch + collection-scoped search"
```

---

## Task 8: Wire `app_controller.py` to use dispatch + auto-pick `source_dir`

**Files:**
- Modify: `src/apis/app_controller.py`

- [ ] **Step 1: Update imports**

Open `src/apis/app_controller.py`. Replace lines 16 and 20:

```python
from src.constants.app_constant import PDFS_DIR, COLLECTION_NAME, T_ZIP, TT_ZIP, STORAGE_FAQ_TEMPLATE
```
```python
from src.rag.search.pipeline import async_pipeline_hyde_search
```

with:

```python
from src.constants.app_constant import (
    PDFS_DIR, COLLECTION_NAME, T_ZIP, TT_ZIP, STORAGE_FAQ_TEMPLATE,
    OVERALL_COLLECTION_NAME, OVERALL_INGEST_DIR, INGEST_DIR,
)
```
```python
from src.rag.search.pipeline import async_pipeline_dispatch
```

- [ ] **Step 2: Update `/ingest` endpoint to auto-pick `source_dir`**

Find the function `ingest_documents` (lines 24-46). Replace its body with:

```python
@documents_router.get("/ingest")
async def ingest_documents(force_restart: bool = False, collection_name: str | None = None,
                           background_tasks: BackgroundTasks = None):
    target_collection = collection_name or COLLECTION_NAME
    source_dir = (
        OVERALL_INGEST_DIR
        if target_collection == OVERALL_COLLECTION_NAME
        else INGEST_DIR
    )
    try:
        if not force_restart:
            manager = get_qdrant_client(target_collection)
            exists = await manager.async_client.collection_exists(target_collection)
            if exists:
                info = await manager.async_client.get_collection(target_collection)
                if info.points_count and info.points_count > 0:
                    return {
                        "status": "skipped",
                        "message": f"Collection '{target_collection}' already exists with {info.points_count} points. Use force_restart=true to re-ingest.",
                        "points_count": info.points_count,
                    }
    except Exception as exc:
        logger.exception("Ingest failed")
        raise QdrantApiError(f"Ingest failed: {exc}") from exc

    background_tasks.add_task(
        upload_to_qdrant,
        force_restart=force_restart,
        collection_name=target_collection,
        source_dir=source_dir,
    )
    return {"status": "started",
            "message": f"Ingest triggered in background for collection '{target_collection}' from '{source_dir}'."}
```

Key changes:
- Auto-picks `source_dir` based on `target_collection`.
- `manager = get_qdrant_client(target_collection)` (no more `manager.collection_name = X` mutation).
- `background_tasks.add_task(...)` now passes `source_dir` kwarg.

- [ ] **Step 3: Update `/chat/ask` to use `async_pipeline_dispatch`**

Find the function `chat_ask` (lines 49-77). Replace line 53:

```python
        result = await async_pipeline_hyde_search(
            user_input=payload.message,
            conversation_id=payload.conversation_id
        )
```

with:

```python
        result = await async_pipeline_dispatch(
            user_input=payload.message,
            conversation_id=payload.conversation_id
        )
```

- [ ] **Step 4: Verify imports**

Run:

```bash
python -c "from src.apis.app_controller import chat_ask, ingest_documents; print('OK')"
```

Expected output: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/apis/app_controller.py
git commit -m "feat: wire app controller to dispatch + auto source_dir for overall"
```

---

## Task 9: End-to-end smoke test

**Files:** None (verification only)

**Prerequisites:** Qdrant must be reachable at `QDRANT_URL` (see `.env`); `OPENAI_*` keys configured.

- [ ] **Step 1: Verify app boots with all 3 routes**

Start the server in one terminal:

```bash
python main.py
```

Expected log lines (look for these):
- `Loaded precomputed embeddings: ['core_knowledge', 'off_topic', 'overall_course_knowledge']` (cache hit)
  OR
- `Encoding 28 samples cho route 'overall_course_knowledge'...` (first build after samples added)
- App starts on `http://0.0.0.0:8083`

- [ ] **Step 2: Trigger ingest for `overall_course` collection**

In a second terminal:

```bash
curl "http://localhost:8083/api/v1/documents/ingest?collection_name=overall_course&force_restart=true"
```

Expected JSON response:
```json
{"status":"started","message":"Ingest triggered in background for collection 'overall_course' from '...\\data\\ingest_overall'."}
```

- [ ] **Step 3: Wait + verify Qdrant collection populated**

Wait ~60-120 seconds (depending on embedding throughput). Then:

```bash
curl "http://localhost:8083/api/v1/documents/ingest?collection_name=overall_course"
```

Expected JSON (status now `skipped` because collection has points):
```json
{"status":"skipped","message":"Collection 'overall_course' already exists with 380 points. Use force_restart=true to re-ingest.","points_count":380}
```

(Number may be ~380 ± a few depending on lesson count in the data.)

- [ ] **Step 4: Smoke-test 5 overall queries**

Run each curl below, verify the answer matches expectation.

```bash
curl -X POST http://localhost:8083/api/v1/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"What does LOMA stand for?","conversation_id":""}'
```
Expected: response mentions "Life Office Management Association"; `intent: "core_knowledge"` (note: `PipelineResult.intent` is set to `INTENT_CORE_KNOWLEDGE` regardless of which collection answered — by design, since the overall topic is still considered "course knowledge").

```bash
curl -X POST http://localhost:8083/api/v1/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"How many modules are in LOMA 281?","conversation_id":""}'
```
Expected: response says "4 modules".

```bash
curl -X POST http://localhost:8083/api/v1/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"What is covered in Module 2 of LOMA 281?","conversation_id":""}'
```
Expected: response lists 4 lessons (Term Life, Cash Value Life, Annuities, Health Insurance).

```bash
curl -X POST http://localhost:8083/api/v1/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"Compare LOMA 281 and LOMA 291.","conversation_id":""}'
```
Expected: response describes both courses with module/lesson counts.

```bash
curl -X POST http://localhost:8083/api/v1/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the definition of insurable interest?","conversation_id":""}'
```
Expected: response gives the definition (this comes from a node-level chunk).

- [ ] **Step 5: Smoke-test core knowledge query (regression check)**

```bash
curl -X POST http://localhost:8083/api/v1/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain adverse selection in detail with examples.","conversation_id":""}'
```
Expected: detailed answer with `sources` populated (PDF file links). Confirms the existing `imt_kb_v1` collection still works after the refactor.

- [ ] **Step 6: Smoke-test off-topic query (regression check)**

```bash
curl -X POST http://localhost:8083/api/v1/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the weather today?","conversation_id":""}'
```
Expected: canned off-topic message in detected language; `intent: "off_topic"`.

- [ ] **Step 7: Inspect routing logs**

Look at server logs from the previous 7 requests. For each, find the line:

```
Intent routing — score: <float>, intent: <intent_name>
```

Verify:
- The 4 overall metadata queries (steps 4) routed to `overall_course_knowledge`.
- The "What is the definition of insurable interest?" query routed to either `core_knowledge` or `overall_course_knowledge` — both are acceptable since node chunks are in the overall collection.
- The "adverse selection" query routed to `core_knowledge`.
- The weather query routed to `off_topic`.

If any query misrouted, note which one. Tuning options (out of scope for this plan): add more samples to the relevant route in `samples.py`, then re-run cache build.

- [ ] **Step 8: Stop server**

`Ctrl+C` in the server terminal.

- [ ] **Step 9: Final commit (if any tweaks were needed during smoke test)**

If routing samples needed tweaking:

```bash
git add src/rag/semantic_router/samples.py
git commit -m "tune: adjust samples after overall topic smoke test"
```

Otherwise no commit needed.

---

## Done

After all 9 tasks pass, the feature is complete:
- New intent `overall_course_knowledge` is registered and routes metadata-style queries.
- Collection `overall_course` (~380 chunks) is populated in Qdrant.
- Pipeline cleanly splits dispatch from collection-scoped search.
- Race condition on `qdrant.collection_name` mutation is fixed.
- Existing `core_knowledge` and `off_topic` paths remain unchanged.
