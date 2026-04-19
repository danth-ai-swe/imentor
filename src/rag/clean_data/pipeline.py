import json
from pathlib import Path
from typing import Callable

from src.constants.app_constant import INGEST_DIR
from src.rag.clean_data.entrypoint import Chunk, run_pipeline_pdf, run_pipeline_md
from src.utils.checkpoint_utils import (
    load_checkpoint,
    mark_completed,
    mark_failed,
)
from src.utils.logger_utils import logger


def chunks_to_ingest_json(chunks: list[Chunk]) -> list[dict]:
    return [
        {
            "text": chunk.text,
            "payload": {
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "page_number": chunk.page_number,
                "total_pages": chunk.total_pages,
                "file_name": chunk.file_name,
                "category": chunk.category,
                "node_name": chunk.node_name,
                "node_id": chunk.node_id,
            },
        }
        for chunk in chunks
    ]


def save_ingest_json(records: list[dict], source_path: str | Path) -> Path:
    INGEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INGEST_DIR / f"{Path(source_path).stem}.json"
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("      Saved ingest JSON: %s  (%d records)", out_path, len(records))
    return out_path


def _run_folder_pipeline(
        *,
        files: list[Path],
        process_fn: Callable[[Path], list[Chunk]],
        retry_failed: bool,
) -> None:
    """
    Vòng lặp chung cho mọi loại folder pipeline.
    - Đọc / ghi checkpoint.
    - Skip file đã done hoặc failed (trừ khi retry_failed=True).
    - Gọi process_fn(file_path) → chunks → format → save.
    """
    checkpoint = load_checkpoint()
    completed_set = set(checkpoint["completed"])
    failed_set = set(checkpoint["failed"].keys())

    for idx, file_path in enumerate(files, start=1):
        filename = file_path.name
        logger.info("\n[%d/%d] %s", idx, len(files), filename)

        if filename in completed_set:
            logger.info("   Đã xử lý thành công trước đó — bỏ qua.")
            continue

        if filename in failed_set and not retry_failed:
            info = checkpoint["failed"][filename]
            logger.warning(
                "   Đã lỗi trước đó (%s) — bỏ qua.\n   Lỗi: %s\n   Dùng retry_failed=True để thử lại.",
                info.get("timestamp", ""),
                info.get("error", "")[:120],
            )
            continue

        try:
            chunks = process_fn(file_path)
            records = chunks_to_ingest_json(chunks)
            save_ingest_json(records, file_path)
            mark_completed(checkpoint, filename)
            logger.info("   Hoàn tất — %d records", len(records))
        except Exception:
            logger.exception("   Lỗi khi xử lý %s", filename)
            mark_failed(checkpoint, filename, str(Exception))


def run_folder_pdf_pipeline(
        pdf_folder: str | Path,
        xlsx_path: str | Path,
        chunk_size: int = 800,
        retry_failed: bool = False,
) -> None:
    """Xử lý toàn bộ *.pdf trong pdf_folder."""
    pdf_folder = Path(pdf_folder)
    xlsx_path = Path(xlsx_path)
    files = sorted(pdf_folder.glob("*.pdf"))
    if not files:
        logger.warning("Không tìm thấy file PDF nào trong: %s", pdf_folder)
        return

    def process(path: Path) -> list[Chunk]:
        return run_pipeline_pdf(str(path), str(xlsx_path), chunk_size)

    _run_folder_pipeline(
        files=files,
        process_fn=process,
        retry_failed=retry_failed,
    )


def run_folder_md_pipeline(
        md_folder: str | Path,
        xlsx_path: str | Path,
        chunk_size: int = 800,
        retry_failed: bool = False,
) -> None:
    """Xử lý toàn bộ *.md trong md_folder."""
    md_folder = Path(md_folder)
    xlsx_path = Path(xlsx_path)
    files = sorted(md_folder.glob("*.md"))
    if not files:
        logger.warning("Không tìm thấy file MD nào trong: %s", md_folder)
        return

    def process(path: Path) -> list[Chunk]:
        return run_pipeline_md(str(path), str(xlsx_path), chunk_size)

    _run_folder_pipeline(
        files=files,
        process_fn=process,
        retry_failed=retry_failed,
    )
