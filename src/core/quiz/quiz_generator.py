"""
Quiz Generator — Dynamic Batch Mode
=====================================
Tự động tính batch_size để tối ưu wall time:
  n ≤ max_concurrent  →  batch_size=1  (all parallel, ~6s)
  n >  max_concurrent →  batch_size tăng, capped MAX_BATCH_SIZE
"""

import asyncio
import json
import math
import random
from functools import lru_cache
from typing import Any

import pandas as pd

from src.config.app_config import get_app_config
from src.constants.app_constant import METADATA_NODE_XLSX
from src.core.quiz.node_sampler import NodeSampler
from src.core.quiz.prompt import QUIZ_SYSTEM_PROMPT
from src.rag.llm.chat_llm import AzureChatClient, get_openai_chat_client
from src.utils.logger_utils import logger, StepTimer

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_CONCURRENT_LLM = 10
MAX_BATCH_SIZE = 5
TOKENS_PER_QUESTION = 600
MAX_TOKENS_PER_CALL = 3_500
DIFFICULTY_LEVELS = ("Beginner", "Intermediate", "Advanced")

# Type aliases
CategoryChunks = dict[str, list[dict[str, Any]]]
SampledItem = tuple[str, list[dict[str, Any]]]
BatchItem = tuple[str, list[dict[str, Any]], list[dict[str, str]]]


# ── Node metadata ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_category_node_metadata() -> dict[str, list[dict[str, str]]]:
    """Load metadata_node.xlsx → {category: [node_info, ...]}. Cached per process."""
    df = pd.read_excel(METADATA_NODE_XLSX, engine="openpyxl")
    result: dict[str, list[dict[str, str]]] = {}
    for _, row in df.iterrows():
        cat = str(row.get("Category", "")).strip()
        if not cat:
            continue
        result.setdefault(cat, []).append({
            "node_name": str(row.get("Node Name", "")).strip(),
            "definition": str(row.get("Definition", "")).strip(),
            "category": cat,
            "related_nodes": str(row.get("Related Nodes", "")).strip(),
        })
    return result


# ── Difficulty distribution ───────────────────────────────────────────────────

def _distribute_into_counts(weights: list[float], n: int) -> list[int]:
    """Chia n thành 3 phần theo weights, đảm bảo tổng đúng bằng n."""
    total_w = sum(weights)
    raw = [w / total_w * n for w in weights]
    counts = [int(r) for r in raw]
    remainder = n - sum(counts)
    # Phân phối phần dư cho các level có fractional part lớn nhất
    for _, idx in sorted(enumerate(raw), key=lambda x: x[1] - int(x[1]), reverse=True)[:remainder]:
        counts[idx] += 1
    return counts


def _compute_difficulty_distribution(
        n: int,
        quiz_type: str = "random",
        rate_value: str | None = None,
) -> dict[str, Any]:
    """Trả về {counts: {level: int}, rate: {level: float}}."""
    if quiz_type == "rate" and rate_value:
        weights = [float(p.strip()) for p in rate_value.split("|")]
    else:
        weights = [random.randint(1, 10) for _ in range(3)]

    counts = _distribute_into_counts(weights, n)
    return {
        "counts": {DIFFICULTY_LEVELS[i]: counts[i] for i in range(3)},
        "rate": {
            DIFFICULTY_LEVELS[i]: round(counts[i] / n * 100, 1) if n else 0
            for i in range(3)
        },
    }


def _build_difficulty_instruction(counts: dict[str, int]) -> str:
    lines = [f"   - {diff}: exactly {cnt} question(s)" for diff, cnt in counts.items() if cnt > 0]
    return "DIFFICULTY DISTRIBUTION (you MUST follow this exactly):\n" + "\n".join(lines)


# ── Prompt building ───────────────────────────────────────────────────────────

def _build_batch_user_prompt(batch: list[BatchItem]) -> str:
    parts: list[str] = []
    for idx, (category, chunks, node_meta_list) in enumerate(batch, 1):
        parts.append(f"═══ Category Group {idx}: {category} ═══")

        if node_meta_list:
            parts.append(f"[Knowledge Graph Nodes for '{category}']")
            for nm in node_meta_list:
                parts.append(
                    f"  • Node Name: {nm['node_name']}\n"
                    f"    Definition: {nm['definition']}\n"
                    f"    Category: {nm['category']}\n"
                    f"    Related Nodes: {nm['related_nodes']}"
                )
            parts.append("")

        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"--- Chunk {i} ---\n"
                + "\n".join(f"{k}: {chunk.get(k, '')}" for k in (
                    "file_name", "node_name", "category",
                    "page_number", "total_pages", "module", "lesson",
                ))
                + f"\ntext:\n{chunk.get('text', '')}"
            )
        parts.append("")

    parts.append(
        f"Generate exactly {len(batch)} multiple-choice question(s), "
        "one per category group above."
    )
    return "\n".join(parts)


# ── LLM call ──────────────────────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> list[dict]:
    """Parse LLM response (JSON mode hoặc markdown-fenced JSON) → list[dict]."""
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(cleaned)
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    return data if isinstance(data, list) else [data]


async def _agenerate_batch(
        llm: AzureChatClient,
        batch: list[BatchItem],
        semaphore: asyncio.Semaphore,
        batch_idx: int,
        difficulty_instruction: str = "",
) -> list[dict[str, Any]]:
    n_questions = len(batch)
    async with semaphore:
        logger.info(
            f"  🚀 Batch {batch_idx}: {n_questions} question(s) for "
            f"{[cat for cat, _, _ in batch]}"
        )
        system_prompt = QUIZ_SYSTEM_PROMPT.format(
            n_questions=n_questions,
            difficulty_instruction=difficulty_instruction,
        )
        raw = await llm.acreate_json_message(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": _build_batch_user_prompt(batch)}],
            max_tokens=min(n_questions * TOKENS_PER_QUESTION, MAX_TOKENS_PER_CALL),
        )
        questions = _parse_llm_json(raw)
        logger.info(f"  ✅ Batch {batch_idx}: got {len(questions)} question(s)")
        return questions


# ── Source enrichment ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_base_url() -> str:
    config = get_app_config()
    return config.APP_DOMAIN if config.PROFILE_NAME == "prod" \
        else f"http://{config.APP_IP}:{config.APP_PORT}"


def _build_chunk_meta_index(chunks: list[dict[str, Any]]) -> dict[str, dict]:
    """file_name → {module, lesson, course, file_name}"""
    index: dict[str, dict] = {}
    for c in chunks:
        fn = c.get("file_name", "")
        if fn and fn not in index:
            index[fn] = {k: c.get(k) for k in ("module", "lesson", "course", "file_name")}
    return index


def _match_file(source_name: str, chunk_meta: dict[str, dict]) -> str | None:
    """Tìm file_name khớp với source_name (substring match)."""
    for fn in chunk_meta:
        if fn in source_name or source_name in fn:
            return fn
    return next(iter(chunk_meta), None)  # fallback to first


def _enrich_sources(question: dict[str, Any], chunks: list[dict[str, Any]]) -> None:
    """Gán module, lesson, course, url cho từng source trong question."""
    base_url = _get_base_url()
    chunk_meta = _build_chunk_meta_index(chunks)

    for source in question.get("sources", []):
        matched = _match_file(source.get("name", ""), chunk_meta)
        if matched:
            meta = chunk_meta[matched]
            source.update({
                "name": matched,
                "module": meta["module"],
                "lesson": meta["lesson"],
                "course": meta["course"],
                "url": f"{base_url}/api/v1/file/{matched}.pdf",
            })
        else:
            source.setdefault("module", None)
            source.setdefault("lesson", None)
            source.setdefault("course", None)


def _enrich_question_sources(
        question: dict[str, Any],
        category_chunks: CategoryChunks,
) -> None:
    cat = question.get("category", "")
    chunks = category_chunks.get(cat) or [c for cc in category_chunks.values() for c in cc]
    _enrich_sources(question, chunks)


# ── Batch sizing ──────────────────────────────────────────────────────────────

def _compute_batch_size(n: int, max_concurrent: int) -> int:
    """Tính batch_size tối ưu để minimize số LLM rounds."""
    if n <= max_concurrent:
        return 1
    return min(math.ceil(n / max_concurrent), MAX_BATCH_SIZE)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _assign_difficulties(
        sampled_items: list[SampledItem],
        distribution: dict[str, Any],
) -> list[tuple[str, list[dict], str]]:
    """Gán difficulty ngẫu nhiên cho từng sampled item theo distribution."""
    diff_sequence: list[str] = []
    for diff, cnt in distribution["counts"].items():
        diff_sequence.extend([diff] * cnt)
    random.shuffle(diff_sequence)

    return [
        (cat, chunks, diff_sequence[i] if i < len(diff_sequence) else random.choice(DIFFICULTY_LEVELS))
        for i, (cat, chunks) in enumerate(sampled_items)
    ]


def _build_batch_tasks(
        items_with_diff: list[tuple[str, list[dict], str]],
        batch_size: int,
        llm: AzureChatClient,
        semaphore: asyncio.Semaphore,
        category_node_meta: dict[str, list[dict[str, str]]],
) -> list:
    """Chia items thành batches và tạo LLM tasks."""
    batches = [
        items_with_diff[i: i + batch_size]
        for i in range(0, len(items_with_diff), batch_size)
    ]
    tasks = []
    for idx, batch_items in enumerate(batches, 1):
        batch_counts: dict[str, int] = {}
        for _, _, diff in batch_items:
            batch_counts[diff] = batch_counts.get(diff, 0) + 1

        batch_tuples: list[BatchItem] = [
            (cat, chunks, category_node_meta.get(cat, []))
            for cat, chunks, _ in batch_items
        ]
        tasks.append((
            batch_items,
            _agenerate_batch(
                llm=llm,
                batch=batch_tuples,
                semaphore=semaphore,
                batch_idx=idx,
                difficulty_instruction=_build_difficulty_instruction(batch_counts),
            )
        ))
    return tasks


def _assemble_questions(
        batch_tasks: list,
        category_chunks: CategoryChunks,
) -> tuple[list[dict], int, int]:
    """Gom kết quả từ batches, enrich sources, trả về (questions, success, fail)."""
    questions: list[dict] = []
    success = fail = 0
    for batch_items, result in batch_tasks:
        if isinstance(result, Exception):
            fail += len(batch_items)
            logger.warning(f"  ❌ Batch failed: {result}")
            continue
        for q in result:
            q["id"] = len(questions) + 1
            _enrich_question_sources(q, category_chunks)
            questions.append(q)
            success += 1
    return questions, success, fail


async def generate_quiz(
        knowledge_pack: str = "LOMA281",
        level: str | None = None,
        level_value: str | None = None,
        quiz_type: str = "random",
        rate_value: str | None = None,
        n: int = 10,
        window: int = 2,
        max_concurrent: int = MAX_CONCURRENT_LLM,
) -> dict[str, Any]:
    """End-to-end quiz generation pipeline."""
    timer = StepTimer("generate_quiz")
    semaphore = asyncio.Semaphore(max_concurrent)

    try:
        # Step 1: Sample chunks
        async with timer.astep("sample_and_fetch_chunks"):
            sampler = NodeSampler()
            sampled_items = await sampler.asample_and_fetch(
                knowledge_pack=knowledge_pack,
                level=level,
                value=level_value,
                n=n,
                window=window,
            )

        category_chunks: CategoryChunks = {}
        for cat, chunks in sampled_items:
            category_chunks.setdefault(cat, []).extend(chunks)

        # Step 2: Difficulty + metadata
        actual_n = len(sampled_items)
        distribution = _compute_difficulty_distribution(actual_n, quiz_type, rate_value)
        logger.info(f"  🎯 Difficulty: {distribution['counts']}")

        category_node_meta = _load_category_node_metadata()
        items_with_diff = _assign_difficulties(sampled_items, distribution)

        # Step 3: Batch + LLM calls
        batch_size = _compute_batch_size(actual_n, max_concurrent)
        logger.info(
            f"  📋 {actual_n} items → batch_size={batch_size}, "
            f"max_concurrent={max_concurrent}"
        )

        async with timer.astep("generate_questions_batched"):
            llm = get_openai_chat_client()
            raw_tasks = _build_batch_tasks(
                items_with_diff, batch_size, llm, semaphore, category_node_meta
            )
            # Tách coroutines để gather
            batch_items_list = [b for b, _ in raw_tasks]
            coros = [coro for _, coro in raw_tasks]
            batch_results = await asyncio.gather(*coros, return_exceptions=True)

        # Step 4: Assemble
        async with timer.astep("assemble_and_enrich"):
            paired = list(zip(batch_items_list, batch_results))
            questions, success_count, fail_count = _assemble_questions(paired, category_chunks)
            logger.info(
                f"  📊 {success_count} succeeded, {fail_count} failed → "
                f"{len(questions)} questions total"
            )

        return {
            "success": True,
            "data": {
                "total": len(questions),
                "knowledge_pack": knowledge_pack,
                "level": level,
                "level_value": level_value,
                "type": quiz_type,
                "difficulty_distribution": distribution,
                "questions": questions,
            },
        }
    finally:
        timer.summary()
