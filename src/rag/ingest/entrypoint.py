import asyncio
import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
from qdrant_client import models

from src.constants.app_constant import (
    CLASSIFY_WORKERS,
    DEFAULT_CLEANED_DIR,
    DOCKER_CLEANED_DIR, INGEST_DIR,
)
from src.constants.app_constant import COLLECTION_NAME
from src.rag.chunking.agentic_chunker import get_agentic_chunker
from src.rag.db_vector import get_qdrant_client
from src.rag.ingest.prompt import CLASSIFIER_PROMPT_TEMPLATE
from src.rag.llm.chat_llm import AzureChatClient, get_openai_chat_client
from src.rag.load_document.md_loader import load_markdown_file
from src.utils.checkpoint_utils import (
    load_checkpoint as _load_checkpoint_raw,
    save_checkpoint as _save_checkpoint_raw,
    clear_checkpoint,
)
from src.utils.logger_utils import logger, StepTimer


def load_checkpoint() -> set[str]:
    """Wrapper: trả về set completed files (tương thích code cũ của ingest)."""
    data = _load_checkpoint_raw()
    return set(data["completed"])


def save_checkpoint(completed: set[str]) -> None:
    """Wrapper: nhận set completed, ghi theo format chung."""
    data = _load_checkpoint_raw()
    data["completed"] = sorted(completed)
    _save_checkpoint_raw(data)


def parse_source_filename(file_name: str) -> Dict[str, str]:
    stem = Path(file_name).stem
    parts = stem.split("_")

    course = parts[0].strip() if len(parts) >= 1 else "Unknown"
    module_lesson = parts[1].strip() if len(parts) >= 2 else "Unknown"

    module = "Unknown"
    lesson = "Unknown"
    match = re.fullmatch(r"(M\d+)(L\d+)", module_lesson)
    if match:
        module, lesson = match.groups()

    remaining_parts = parts[2:] if len(parts) > 2 else []
    remaining_name = re.sub(r"^Knowledge File_?", "", "_".join(remaining_parts)).strip()
    document_name = remaining_name or stem

    return {
        "module_lesson": module_lesson,
        "document_name": document_name,
        "heading_path": f"{course} > {module} > {lesson}",
        "page_id": f"{course}_{module_lesson}_{document_name}".replace(" ", "_"),
    }


def load_node_metadata(node_xlsx_path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        df = pd.read_excel(node_xlsx_path)
        metadata: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            node_id = str(row.get("Node ID", "")).strip()
            if node_id:
                metadata[node_id] = {
                    "node_id": node_id,
                    "node_name": str(row.get("Node Name", "")).strip(),
                    "category": str(row.get("Category", "")).strip(),
                    "definition": str(row.get("Definition", "")).strip(),
                    "domain_tags": str(row.get("Domain Tags", "")).strip(),
                    "source": str(row.get("Source", "")).strip(),
                    "related_nodes": str(row.get("Related Nodes", "")).strip(),
                }
        return metadata
    except Exception as e:
        logger.error(f"❌ Failed to load {node_xlsx_path}: {e}")
        return {}


def build_node_options(node_metadata: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "node_id": str(info.get("node_id", "")).strip(),
            "node_name": str(info.get("node_name", "")).strip(),
            "category": str(info.get("category", "")).strip(),
        }
        for info in node_metadata.values()
        if all([info.get("node_id"), info.get("node_name"), info.get("category")])
    ]


def chunk_text(text: str, chunker) -> List[str]:
    if not text or len(text.strip()) < 100:
        return []
    try:
        chunks = chunker.split_text(text)
        return [c.strip() for c in chunks if c.strip()]
    except Exception as e:
        logger.warning(f"⚠️  Chunking failed: {e}")
        return [text]


def build_chunk_document(
        chunk_index: int,
        chunk_text: str,
        category: str,
        node_id: str,
        file_info: Dict[str, str],
        source_file: str,
        total_chunks: int,
        char_start: int,
        char_end: int,
        run_ts: str,
) -> Dict[str, Any]:
    return {
        "text": chunk_text,
        "payload": {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "page_id": file_info["page_id"],
            "total_pages": 1,
            "source_file": source_file,
            "document_name": file_info["document_name"],
            "heading_path": file_info["heading_path"],
            "char_start": char_start,
            "char_end": char_end,
            "file_type": "markdown",
            "category": category,
            "node_id": node_id,
            "timestamp": run_ts,
        },
    }


def discover_markdown_files(root_dir: Path) -> List[Path]:
    root_dir = root_dir.expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        logger.warning(f"⚠️  Cleaned data directory not found: {root_dir}")
        return []
    return sorted(root_dir.rglob("*.md"), key=lambda p: str(p).lower())


def find_node_file_for_markdown(md_file: Path) -> Optional[Path]:
    candidates = [
        md_file.with_name(md_file.name.replace("_Knowledge File", "_Knowledge Node").replace(".md", ".xlsx")),
        md_file.with_name(md_file.name.replace("Knowledge File", "Knowledge Node").replace(".md", ".xlsx")),
    ]

    stem_parts = md_file.stem.split("_")
    if len(stem_parts) >= 2:
        prefix = f"{stem_parts[0]}_{stem_parts[1]}"
        candidates += sorted(md_file.parent.glob(f"{prefix}*Knowledge Node*.xlsx"))

    candidates += sorted(md_file.parent.glob("*Knowledge Node*.xlsx"))
    candidates += sorted(md_file.parent.glob("*.xlsx"))

    seen: set = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


_thread_local = threading.local()


def get_cleaned_data_dir() -> Path:
    env_dir = os.getenv("IMT_CLEANED_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    if DEFAULT_CLEANED_DIR.exists():
        return DEFAULT_CLEANED_DIR

    return DOCKER_CLEANED_DIR


def _get_thread_llm_client() -> AzureChatClient:
    if not hasattr(_thread_local, "llm_client"):
        _thread_local.llm_client = AzureChatClient()
    return _thread_local.llm_client


# ══════════════════════════════════════════════════════════════════════════════
# Async classify chunk
# ══════════════════════════════════════════════════════════════════════════════
async def aclassify_chunk(
        chunk_text_val: str,
        node_options: List[Dict[str, str]],
        llm_client: AzureChatClient,
        max_retries: int = 3,
) -> Tuple[str, str, str]:
    """Async version of classify_chunk using llm.ainvoke."""
    if not node_options:
        return "Unknown", "Unknown", "N/A"

    candidates = "\n".join(
        f"- node_id={o['node_id']} | node_name={o['node_name']} | category={o['category']}"
        for o in node_options
    )
    prompt = CLASSIFIER_PROMPT_TEMPLATE.format(candidates=candidates, chunk_text=chunk_text_val)

    backoff = 2.0
    last_exc: Exception = RuntimeError("aclassify_chunk: no attempts made")

    for attempt in range(1, max_retries + 1):
        try:
            response = (await llm_client.ainvoke(prompt)).strip()
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            parsed = json.loads(json_match.group(0) if json_match else response)

            selected_category = str(parsed.get("category", "")).strip()
            selected_node_name = str(parsed.get("node_name", "")).strip()

            for item in node_options:
                if item["category"] == selected_category and item["node_name"] == selected_node_name:
                    return item["category"], item["node_name"], item["node_id"]
            for item in node_options:
                if item["node_name"] == selected_node_name:
                    return item["category"], item["node_name"], item["node_id"]
            for item in node_options:
                if item["category"] == selected_category:
                    return item["category"], item["node_name"], item["node_id"]

            fb = node_options[0]
            return fb["category"], fb["node_name"], fb["node_id"]

        except (json.JSONDecodeError, Exception) as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    f"    ⚠️  aclassify attempt {attempt}/{max_retries} failed: {exc} — retry in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 15)

    logger.error(f"    ❌ aclassify failed after {max_retries} attempts: {last_exc} — using fallback")
    fb = node_options[0]
    return fb["category"], fb["node_name"], fb["node_id"]


# ══════════════════════════════════════════════════════════════════════════════
# Async ingest pipeline
# ══════════════════════════════════════════════════════════════════════════════
async def async_run_ingest_pipeline(force_restart: bool = False) -> None:
    """Fully async ingest pipeline using AsyncQdrantClient + async LLM."""
    run_ts = pd.Timestamp.now().isoformat()
    logger.info("🚀 Starting ASYNC FULL INGEST Pipeline...")

    # ── Checkpoint ───────────────────────────────────────────────────────────
    if force_restart:
        clear_checkpoint()
    completed_files: set[str] = load_checkpoint()
    if completed_files:
        logger.info(f"🔖 Resuming — {len(completed_files)} file(s) already done, will skip them.")

    # ── Step 1: Discover files ───────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("📂 [Step 1] Discover .md files + match node.xlsx")

    cleaned_dir = get_cleaned_data_dir()
    logger.info(f"   📁 {cleaned_dir}")

    md_files = discover_markdown_files(cleaned_dir)
    if not md_files:
        logger.error("❌ No markdown files found. Exiting.")
        return

    file_node_map: Dict[Path, Optional[Path]] = {
        md: find_node_file_for_markdown(md) for md in md_files
    }
    matched = sum(1 for v in file_node_map.values() if v)
    logger.info(
        f"   ✅ {len(md_files)} md files, {matched}/{len(md_files)} node.xlsx matched"
    )

    # ── Init Qdrant (async) ──────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("🔧 [Setup] Qdrant collection (async)")
    manager = get_qdrant_client()
    manager.collection_name = COLLECTION_NAME
    await manager.acreate_collection(recreate=force_restart)
    await manager.acreate_payload_index("category", models.PayloadSchemaType.KEYWORD)
    await manager.acreate_payload_index("document_name", models.PayloadSchemaType.KEYWORD)
    await manager.acreate_payload_index("learning_path", models.PayloadSchemaType.KEYWORD)

    # ── Shared resources ─────────────────────────────────────────────────────
    chunker = get_agentic_chunker()
    llm = get_openai_chat_client()

    # ── Per-file loop ─────────────────────────────────────────────────────────
    total_chunk_count = skipped_files = resumed_files = 0

    for file_idx, md_file in enumerate(md_files, 1):
        logger.info("─" * 60)
        logger.info(f"📄 [{file_idx}/{len(md_files)}] {md_file.name}")

        # ── Resume: skip already-completed files ─────────────────────────────
        if md_file.stem in completed_files:
            logger.info("   ⏭️  Already completed — skipping.")
            resumed_files += 1
            continue

        node_file = file_node_map[md_file]

        # ── Step 2: Load metadata ─────────────────────────────────────────────
        node_meta = load_node_metadata(node_file) if node_file else {}
        node_options = build_node_options(node_meta)
        file_info = parse_source_filename(md_file.name)
        logger.info(f"   📋 [Step 2] Metadata: {len(node_options)} nodes")

        # ── Step 3: Chunk (CPU-bound — run in thread) ────────────────────────
        md_content = await asyncio.to_thread(load_markdown_file, md_file)
        chunks = chunk_text(md_content, chunker) if md_content else []
        logger.info(f"   ✂️  [Step 3] Chunks: {len(chunks)}")

        if not chunks:
            logger.warning("   ⚠️  No chunks — skipping file.")
            skipped_files += 1
            completed_files.add(md_file.stem)
            save_checkpoint(completed_files)
            continue

        # ── Step 4: Classify each chunk (async concurrency) ──────────────────
        logger.info(f"   🤖 [Step 4] Classifying {len(chunks)} chunks (async, {CLASSIFY_WORKERS} concurrency)…")

        semaphore = asyncio.Semaphore(CLASSIFY_WORKERS)

        async def _classify_with_sem(idx: int, text: str) -> Tuple[int, str, str]:
            async with semaphore:
                cat, _, nid = await aclassify_chunk(text, node_options, llm)
                return idx, cat, nid

        classify_tasks = [_classify_with_sem(i, c) for i, c in enumerate(chunks)]
        classify_results_list: List[Tuple[str, str]] = [("Unknown", "N/A")] * len(chunks)

        results = await asyncio.gather(*classify_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"      ⚠️  classify failed: {result}")
            else:
                idx, category, node_id = result
                classify_results_list[idx] = (category, node_id)
                logger.debug(f"      chunk {idx + 1}/{len(chunks)}: {category} / {node_id}")

        # ── Step 5: Build metadata ───────────────────────────────────────────
        char_offset = 0
        chunk_data_list: List[Dict[str, Any]] = []

        for chunk_idx, chunk in enumerate(chunks):
            category, node_id = classify_results_list[chunk_idx]
            char_end = char_offset + len(chunk)
            chunk_data_list.append(build_chunk_document(
                chunk_index=chunk_idx,
                chunk_text=chunk,
                category=category,
                node_id=node_id,
                file_info=file_info,
                source_file=md_file.name,
                total_chunks=len(chunks),
                char_start=char_offset,
                char_end=char_end,
                run_ts=run_ts,
            ))
            char_offset = char_end

        # ── Step 6: Upload (async, with per-file retry) ──────────────────────
        logger.info(f"   📤 [Step 6] Uploading {len(chunk_data_list)} docs (async)…")
        upload_ok = False
        file_backoff = 120.0
        _FILE_UPLOAD_MAX_RETRIES = 3

        for attempt in range(1, _FILE_UPLOAD_MAX_RETRIES + 1):
            try:
                await manager.aupload_documents(
                    chunk_data_list,
                    batch_size=50,
                    max_retries=3,
                )
                upload_ok = True
                break
            except Exception as e:
                if attempt < _FILE_UPLOAD_MAX_RETRIES:
                    logger.warning(
                        f"   ⚠️  Upload attempt {attempt}/{_FILE_UPLOAD_MAX_RETRIES} failed: {e}"
                        f" — retry in {file_backoff:.0f}s"
                    )
                    await asyncio.sleep(file_backoff)
                    file_backoff = min(file_backoff * 2, 600)
                else:
                    logger.error(
                        f"   ❌ Upload failed after {_FILE_UPLOAD_MAX_RETRIES} attempts: {e}",
                        exc_info=True,
                    )

        if upload_ok:
            total_chunk_count += len(chunk_data_list)
            completed_files.add(md_file.stem)
            save_checkpoint(completed_files)
        else:
            logger.error(f"   ❌ File {md_file.name} failed permanently — NOT marked in checkpoint.")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("📊  EXECUTION SUMMARY (ASYNC)")
    logger.info("=" * 60)
    logger.info(f"  Files total              {len(md_files):>6}")
    logger.info(f"  Files resumed (skipped)  {resumed_files:>6}")
    logger.info(f"  Files skipped (empty)    {skipped_files:>6}")
    logger.info(f"  Files processed          {len(md_files) - resumed_files - skipped_files:>6}")
    logger.info(f"  Total chunks indexed     {total_chunk_count:>6}")
    logger.info("=" * 60)
    logger.info("🎉 ASYNC PIPELINE COMPLETE!")


async def upload_to_qdrant(force_restart: bool = False, collection_name: str | None = None):
    target_collection = collection_name or COLLECTION_NAME
    timer = StepTimer("ingest_upload")

    try:
        async with timer.astep("prepare_checkpoint"):
            if force_restart:
                clear_checkpoint()

            uploaded_files = load_checkpoint()
            all_json_files = sorted(INGEST_DIR.glob("*.json"))
            pending_files = [f for f in all_json_files if f.name not in uploaded_files]

        logger.info(f"📊 Tổng file: {len(all_json_files)}")
        logger.info(f"✅ Đã upload:  {len(uploaded_files)}")
        logger.info(f"⏳ Còn lại:    {len(pending_files)}\n")

        if not pending_files:
            logger.info("🎉 Tất cả file đã được upload rồi!")
            return

        async with timer.astep("create_collection_and_indexes"):
            manager = get_qdrant_client()
            manager.collection_name = target_collection
            await manager.acreate_collection(recreate=force_restart)
            await manager.acreate_payload_index("category", models.PayloadSchemaType.KEYWORD)
            await manager.acreate_payload_index("file_name", models.PayloadSchemaType.KEYWORD)

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
