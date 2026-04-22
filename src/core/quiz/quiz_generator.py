import asyncio
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from qdrant_client import models

from src.constants.app_constant import METADATA_NODE_XLSX, QUIZ_DIR, DATA_DIR
from src.core.quiz.prompt import QUIZ_SYSTEM_PROMPT
from src.rag.db_vector import get_qdrant_client
from src.rag.llm.chat_llm import get_openai_chat_client
from src.utils.app_utils import parse_llm_json
from src.utils.logger_utils import logger, StepTimer

MAX_QUESTIONS_PER_PROMPT = 10
TOKENS_PER_QUESTION = 600
MAX_TOKENS_PER_CALL = 3_500
MAX_WORKERS = 5

DIFFICULTY_LEVELS = ("Beginner", "Intermediate", "Advanced")
QUIZ_XLSX_DIR = Path(QUIZ_DIR)
QUIZ_OUTPUT_DIR = Path(DATA_DIR)


def _get_output_path(knowledge_pack: str) -> Path:
    """Trả về path file JSON output cho từng knowledge_pack."""
    return QUIZ_OUTPUT_DIR / f"quiz_{knowledge_pack}.json"


def _append_questions_to_file(knowledge_pack: str, questions: list[dict]) -> None:
    """Append danh sách câu hỏi vào file JSON (thread-safe với file lock đơn giản)."""
    if not questions:
        return

    path = _get_output_path(knowledge_pack)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Đọc existing data
    existing: list[dict] = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []

    existing.extend(questions)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    logger.info(f"  💾 Đã ghi {len(questions)} câu vào {path} (tổng: {len(existing)})")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — random module / lesson
# ══════════════════════════════════════════════════════════════════════════════

def _pick_random_module(knowledge_pack: str, df: pd.DataFrame) -> str:
    modules = (
        df[df["Course"].astype(str).str.strip() == knowledge_pack]["Module"]
        .astype(str).str.strip().unique().tolist()
    )
    if not modules:
        raise ValueError(f"Không tìm thấy module nào cho course='{knowledge_pack}'")
    chosen = random.choice(modules)
    logger.info(f"  🎲 Random module → '{chosen}'")
    return chosen


def _pick_random_lesson(knowledge_pack: str, module_value: str, df: pd.DataFrame) -> str:
    lessons = (
        df[
            (df["Course"].astype(str).str.strip() == knowledge_pack) &
            (df["Module"].astype(str).str.strip() == module_value)
            ]["Lesson"]
        .astype(str).str.strip().unique().tolist()
    )
    if not lessons:
        raise ValueError(
            f"Không tìm thấy lesson nào cho course='{knowledge_pack}', module='{module_value}'"
        )
    chosen = random.choice(lessons)
    logger.info(f"  🎲 Random lesson → '{chosen}'")
    return chosen


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & filter metadata_node.xlsx
# ══════════════════════════════════════════════════════════════════════════════

def _load_metadata_rows(
        knowledge_pack: str,
        module_value: str | None = None,
        lesson_value: str | None = None,
) -> list[dict]:
    df = pd.read_excel(METADATA_NODE_XLSX, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    mask = df["Course"].astype(str).str.strip() == knowledge_pack
    if module_value:
        mask &= df["Module"].astype(str).str.strip() == module_value
    if lesson_value:
        mask &= df["Lesson"].astype(str).str.strip() == lesson_value

    filtered = df[mask].copy()
    if filtered.empty:
        detail = f"course='{knowledge_pack}'"
        if module_value:
            detail += f", module='{module_value}'"
        if lesson_value:
            detail += f", lesson='{lesson_value}'"
        raise ValueError(f"Không tìm thấy row nào với {detail}")

    logger.info(f"  📄 metadata rows matched: {len(filtered)}")
    return filtered.to_dict(orient="records")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Phân phối số câu hỏi cho mỗi row
# ══════════════════════════════════════════════════════════════════════════════

def _distribute_counts(total: int, n_rows: int) -> list[int]:
    if total <= n_rows:
        counts = [1] * total
    else:
        per_row = math.ceil(total / n_rows)
        counts = [per_row] * n_rows
    logger.info(
        f"  🔢 {total} câu / {n_rows} rows → "
        f"dùng {len(counts)} row(s), mỗi row sinh {counts[0]} câu"
    )
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Fetch chunks từ vector DB (thuần async)
# ══════════════════════════════════════════════════════════════════════════════

async def _fetch_chunks_for_row(row: dict) -> list[dict]:
    qdrant = get_qdrant_client()

    def _val(key: str) -> str:
        return str(row.get(key, "")).strip()

    conditions = [
        models.FieldCondition(key="course", match=models.MatchValue(value=_val("Course"))),
        models.FieldCondition(key="category", match=models.MatchValue(value=_val("Category"))),
        models.FieldCondition(key="node_name", match=models.MatchValue(value=_val("Node Name"))),
    ]
    scroll_filter = models.Filter(must=conditions)

    points = await qdrant.ascroll_all(scroll_filter=scroll_filter)

    chunks = [
        {
            "text": pt.payload.get("text", ""),
            "chunk_index": pt.payload.get("chunk_index"),
            "total_chunks": pt.payload.get("total_chunks"),
            "page_number": pt.payload.get("page_number"),
            "total_pages": pt.payload.get("total_pages"),
            "file_name": pt.payload.get("file_name", ""),
            "category": pt.payload.get("category", ""),
            "node_name": pt.payload.get("node_name", ""),
            "node_id": pt.payload.get("node_id"),
            "module": pt.payload.get("module", ""),
            "lesson": pt.payload.get("lesson", ""),
            "course": pt.payload.get("course", ""),
        }
        for pt in points
    ]
    chunks.sort(key=lambda c: c.get("chunk_index") or 0)

    logger.info(f"  🔍 [{_val('Node Name')}] → {len(chunks)} chunk(s) từ vector DB")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Load file xlsx ví dụ + filter theo difficulty
# ══════════════════════════════════════════════════════════════════════════════

def _build_xlsx_path(source: str) -> Path:
    name = str(source).strip()
    if not name.lower().endswith(".xlsx"):
        name += ".xlsx"
    return QUIZ_XLSX_DIR / name


def _load_example_questions(source: str, difficulty: str) -> list[dict]:
    path = _build_xlsx_path(source)
    if not path.exists():
        logger.warning(f"  ⚠️  Không tìm thấy file ví dụ: {path}")
        return []

    df = pd.read_excel(path, engine="openpyxl", header=0, skiprows=[1])
    df.columns = [str(c).strip() for c in df.columns]

    diff_col = next(
        (c for c in df.columns if "dif" in c.lower() and "level" in c.lower()), None
    )
    if diff_col is None:
        logger.warning(f"  ⚠️  Không có cột 'Dificulty Level' trong {path.name}")
        return []

    filtered = df[df[diff_col].astype(str).str.strip() == difficulty]
    logger.info(f"  📖 [{path.name}] ví dụ difficulty='{difficulty}': {len(filtered)} câu")
    return filtered.to_dict(orient="records")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Build prompt và gọi LLM
# ══════════════════════════════════════════════════════════════════════════════

def _build_user_prompt(
        row: dict,
        chunks: list[dict],
        n_questions: int,
        difficulty: str,
        examples: list[dict],
) -> str:
    parts: list[str] = []

    parts.append("═" * 60)
    parts.append("KNOWLEDGE NODE (from metadata)")
    parts.append("═" * 60)
    parts.append(f"Course        : {row.get('Course', '')}")
    parts.append(f"Module        : {row.get('Module', '')}")
    parts.append(f"Lesson        : {row.get('Lesson', '')}")
    parts.append(f"Category      : {row.get('Category', '')}")
    parts.append(f"Node Name     : {row.get('Node Name', '')}")
    parts.append(f"Definition    : {row.get('Definition', '')}")
    parts.append(f"Summary       : {row.get('Summary', '')}")
    parts.append(f"Related Nodes : {row.get('Related Nodes', '')}")
    parts.append("")

    parts.append("═" * 60)
    parts.append(f"CONTENT CHUNKS ({len(chunks)} chunk(s))")
    parts.append("═" * 60)
    for chunk in chunks:
        parts.append(
            f"┌─ Chunk {chunk.get('chunk_index')} / {chunk.get('total_chunks')}  "
            f"│  Page {chunk.get('page_number')} / {chunk.get('total_pages')}  "
            f"│  File: {chunk.get('file_name')}"
        )
        parts.append(
            f"│  Course: {chunk.get('course')}  Module: {chunk.get('module')}  "
            f"Lesson: {chunk.get('lesson')}  Category: {chunk.get('category')}  "
            f"Node: {chunk.get('node_name')}"
        )
        parts.append("│")
        parts.append(f"│  {chunk.get('text', '').strip()}")
        parts.append("└" + "─" * 59)
    parts.append("")

    if examples:
        parts.append("═" * 60)
        parts.append(f"EXAMPLE QUESTIONS (difficulty='{difficulty}')")
        parts.append("═" * 60)
        for ex in examples:
            parts.append(f"Q : {ex.get('Question', '')}")
            parts.append(f"  A0: {ex.get('1', '')}")
            parts.append(f"  A1: {ex.get('2', '')}")
            parts.append(f"  A2: {ex.get('3', '')}")
            parts.append(f"  A3: {ex.get('4', '')}")
            parts.append(f"  ✓ Correct index: {ex.get('Correct Answer', '')}")
            parts.append("")
    else:
        parts.append("(No example questions available for this difficulty level.)")
        parts.append("")

    parts.append("═" * 60)
    parts.append("GENERATION REQUEST")
    parts.append("═" * 60)
    parts.append(
        f"Generate exactly {n_questions} multiple-choice question(s) "
        f"at difficulty='{difficulty}' "
        f"for the node '{row.get('Node Name', '')}' above.\n"
        f"Base ALL questions strictly on the chunks provided."
    )
    return "\n".join(parts)


def _call_llm_for_row(
        row: dict,
        chunks: list[dict],
        n_questions: int,
        difficulty: str,
        examples: list[dict],
) -> list[dict]:
    llm = get_openai_chat_client()
    system_prompt = QUIZ_SYSTEM_PROMPT.format(
        n_questions=n_questions,
        difficulty_instruction=(
            f"DIFFICULTY DISTRIBUTION:\n"
            f"   - {difficulty}: exactly {n_questions} question(s)"
        ),
    )
    user_prompt = _build_user_prompt(row, chunks, n_questions, difficulty, examples)
    raw = llm.chat(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=min(n_questions * TOKENS_PER_QUESTION, MAX_TOKENS_PER_CALL),
    )
    return parse_llm_json(raw)


def _enrich_question_metadata(question: dict, row: dict, chunks: list[dict]) -> None:
    question["category"] = str(row.get("Category", "")).strip()
    question["node_name"] = str(row.get("Node Name", "")).strip()

    seen: dict[str, dict] = {}
    for chunk in chunks:
        fn = str(chunk.get("file_name", "")).strip()
        if fn and fn not in seen:
            seen[fn] = chunk

    question["sources"] = [
        {
            "name": fn,
            "url": f"https://api.fpt-apps.com/imt-ai-brain/api/v1/file/{fn}.pdf",
            "page": chunk.get("page_number"),
            "total_pages": chunk.get("total_pages"),
            "module": chunk.get("module") or str(row.get("Module", "")).strip() or None,
            "lesson": chunk.get("lesson") or str(row.get("Lesson", "")).strip() or None,
            "course": chunk.get("course") or str(row.get("Course", "")).strip() or None,
        }
        for fn, chunk in seen.items()
    ]


# ══════════════════════════════════════════════════════════════════════════════
# ASYNC WORKER — xử lý một (row, difficulty), ghi file ngay sau khi xong
# ══════════════════════════════════════════════════════════════════════════════

async def _process_row(
        row: dict,
        n_q: int,
        difficulty: str,
        knowledge_pack: str,
        file_lock: asyncio.Lock,
) -> tuple[list[dict], int, int]:
    node_name = str(row.get("Node Name", "")).strip()
    source = str(row.get("Source", "")).strip()

    logger.info(f"\n  ▶ [{node_name}] difficulty='{difficulty}' — sinh {n_q} câu")

    try:
        chunks = await _fetch_chunks_for_row(row)
    except Exception as e:
        logger.warning(f"  ❌ Fetch chunks thất bại [{node_name}]: {e}")
        return [], 0, n_q

    if not chunks:
        logger.warning(f"  ⚠️  Không có chunk nào cho node '{node_name}', bỏ qua.")
        return [], 0, n_q

    examples = _load_example_questions(source, difficulty) if source else []

    try:
        loop = asyncio.get_event_loop()
        raw_questions = await loop.run_in_executor(
            None,
            lambda: _call_llm_for_row(row, chunks, n_q, difficulty, examples),
        )
    except Exception as e:
        logger.warning(f"  ❌ LLM thất bại [{node_name}] difficulty='{difficulty}': {e}")
        return [], 0, n_q

    result: list[dict] = []
    for q in raw_questions:
        _enrich_question_metadata(q, row, chunks)
        q["difficulty"] = difficulty
        result.append(q)

    # ✅ Ghi file ngay sau khi row xong — dùng lock để tránh concurrent write
    if result:
        async with file_lock:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _append_questions_to_file(knowledge_pack, result),
            )

    return result, len(result), 0


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND TASK — generate_quiz_full chạy nền, ghi file từng row
# ══════════════════════════════════════════════════════════════════════════════

async def generate_quiz_full_background(
        knowledge_pack: str
) -> None:
    """
    Chạy nền: sinh câu hỏi toàn bộ course, append từng batch vào file JSON ngay khi xong.
    Không trả về gì — được gọi qua BackgroundTasks.
    """
    DIFFICULTY_DIST = [
        ("Beginner", 2),
        ("Intermediate", 5),
        ("Advanced", 3),
    ]

    # Xoá file cũ để bắt đầu fresh
    output_path = _get_output_path(knowledge_pack)
    if output_path.exists():
        output_path.unlink()
        logger.info(f"  🗑️  Đã xoá file cũ: {output_path}")

    timer = StepTimer("generate_quiz_full_background")

    with timer.step("load_metadata"):
        df_all = pd.read_excel(METADATA_NODE_XLSX, engine="openpyxl")
        df_all.columns = [c.strip() for c in df_all.columns]

        mask = df_all["Course"].astype(str).str.strip() == knowledge_pack
        filtered = df_all[mask].copy()

        if filtered.empty:
            raise ValueError(f"Không tìm thấy row nào cho course='{knowledge_pack}'")

        rows = filtered.to_dict(orient="records")
        logger.info(f"  📄 Total metadata rows for '{knowledge_pack}': {len(rows)}")

    semaphore = asyncio.Semaphore(MAX_WORKERS)
    # Lock dùng chung cho tất cả task để tránh concurrent write vào file
    file_lock = asyncio.Lock()

    async def _bounded(row, difficulty, n_q):
        async with semaphore:
            return await _process_row(row, n_q, difficulty, knowledge_pack, file_lock)

    with timer.step("generate_per_row"):
        all_results = await asyncio.gather(
            *[
                _bounded(row, difficulty, n_q)
                for row in rows
                for difficulty, n_q in DIFFICULTY_DIST
            ],
        )

    total_success = sum(s for _, s, _ in all_results)
    total_fail = sum(f for _, _, f in all_results)

    logger.info(
        f"\n  📊 [BACKGROUND] Hoàn tất: {total_success} sinh được, {total_fail} thất bại "
        f"→ file: {_get_output_path(knowledge_pack)}"
    )
    timer.summary()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — generate_quiz (async, có filter module/lesson)
# ══════════════════════════════════════════════════════════════════════════════


def generate_quiz(
        knowledge_pack: str,
        total: int,
        difficulty: str | None = None,
        module_value: str | None = None,
        lesson_value: str | None = None,
) -> dict[str, Any]:
    timer = StepTimer("generate_quiz")

    try:
        # ── Pre-load df một lần để dùng cho random ────────────────────────
        df_all = pd.read_excel(METADATA_NODE_XLSX, engine="openpyxl")
        df_all.columns = [c.strip() for c in df_all.columns]

        # Step 0a: Random difficulty nếu không truyền vào
        if not difficulty:
            difficulty = random.choice(DIFFICULTY_LEVELS)
            logger.info(f"  🎲 Random difficulty → '{difficulty}'")

        # Step 0b: Random module nếu không truyền vào
        if not module_value:
            module_value = _pick_random_module(knowledge_pack, df_all)
            # lesson giữ nguyên null — không random khi module chưa được user chỉ định

        # Step 0c: Chỉ random lesson khi module do USER truyền vào nhưng lesson null
        else:
            if not lesson_value:
                lesson_value = _pick_random_lesson(knowledge_pack, module_value, df_all)

        # Step 1: load metadata rows (module + lesson đã được xác định)
        with timer.step("load_metadata"):
            rows = _load_metadata_rows(knowledge_pack, module_value, lesson_value)

        # Step 2: phân phối số câu
        counts = _distribute_counts(total, len(rows))
        rows = rows[:len(counts)]

        max_per_prompt = max(counts) if counts else 0
        if max_per_prompt > MAX_QUESTIONS_PER_PROMPT:
            n_rows = len(rows)
            max_allowed_total = MAX_QUESTIONS_PER_PROMPT * n_rows
            raise ValueError(
                f"Mỗi prompt chỉ được sinh tối đa {MAX_QUESTIONS_PER_PROMPT} câu, "
                f"nhưng hiện tại mỗi prompt cần sinh {max_per_prompt} câu "
                f"({total} câu / {n_rows} node(s)). "
                f"Vui lòng giảm tổng số câu xuống còn tối đa {max_allowed_total} câu."
            )

        questions: list[dict] = []
        success = fail = 0

        # Step 3: gọi LLM song song theo từng row
        with timer.step("generate_per_row"):
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(_process_row, row, n_q, difficulty): idx
                    for idx, (row, n_q) in enumerate(zip(rows, counts))
                }

                results: dict[int, tuple[list[dict], int, int]] = {}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.warning(f"  ❌ Unhandled error trong future (idx={idx}): {e}")
                        results[idx] = ([], 0, counts[idx])

                for idx in range(len(rows)):
                    partial_questions, s, f = results[idx]
                    for q in partial_questions:
                        q["id"] = len(questions) + 1
                        questions.append(q)
                    success += s
                    fail += f

        # Cắt đúng total câu
        questions = questions[:total]
        logger.info(
            f"\n  📊 Kết quả: {success} sinh được, {fail} thất bại "
            f"→ trả về {len(questions)}/{total} câu"
        )

        return {
            "success": True,
            "data": {
                "total": len(questions),
                "knowledge_pack": knowledge_pack,
                "module_value": module_value,
                "lesson_value": lesson_value,
                "difficulty": difficulty,
                "questions": questions,
            },
        }
    finally:
        timer.summary()


def read_quiz_result(knowledge_pack: str) -> dict[str, Any]:
    path = _get_output_path(knowledge_pack)

    if not path.exists():
        return {
            "success": False,
            "message": f"Chưa có kết quả cho '{knowledge_pack}'. File không tồn tại: {path}",
            "data": None,
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            questions = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {
            "success": False,
            "message": f"Lỗi đọc file: {e}",
            "data": None,
        }
    return {
        "success": True,
        "data": {
            "knowledge_pack": knowledge_pack,
            "questions": questions,
        },
    }
