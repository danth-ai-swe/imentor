import re
from pathlib import Path

import fitz

from rag.clean_data.image_processor import ImageProcessor
from rag.clean_data.model import Chunk
from rag.clean_data.node_classifier import NodeClassifier
from src.constants.app_constant import PREPARES_DIR
from src.rag.chunking.recursive_chunker import get_recursive_token_chunk
from src.rag.llm.chat_llm import get_openai_chat_client
from src.utils.app_utils import clean_text
from utils.logger_utils import logger

_PAGE_SEP_RE = re.compile(r"={5} Trang (\d+) / (\d+) ={5}")
_FILENAME_RE = re.compile(r"^(LOMA\d+)_M(\d+)L(\d+)")


def process_pdf(pdf_path: str) -> str:
    """PDF → chuỗi Markdown phân cách theo trang."""
    client = get_openai_chat_client()
    processor = ImageProcessor(client)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    logger.info("PDF: %s  (%d trang)", pdf_path, total_pages)

    sections: list[str] = []

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]
        logger.debug("── Trang %d/%d ──", page_num, total_pages)

        blocks: list[str] = []

        page_text = page.get_text("text").strip()
        if page_text:
            blocks.append(page_text)

        for img_info in page.get_images(full=True):
            result = processor.process_image(doc, xref=img_info[0])
            if result:
                blocks.append(result)

        page_content = (
            "\n\n".join(blocks)
            if blocks
            else "_Trang này không có nội dung văn bản hay ảnh dữ liệu._"
        )
        sections.append(f"===== Trang {page_num} / {total_pages} =====\n\n{page_content}")

    doc.close()
    return "\n\n".join(sections)


def chunk_md_file(md_path: str, chunk_size: int = 800) -> list[Chunk]:
    """Markdown (phân trang) → list[Chunk]."""
    md_path_obj = Path(md_path)
    file_name = md_path_obj.name
    raw = md_path_obj.read_text(encoding="utf-8")

    pages = _split_md_into_pages(raw)
    if not pages:
        raise ValueError(f"Không tìm thấy separator trang trong file: {md_path}")

    total_pages_global = pages[0][1]

    chunker = get_recursive_token_chunk(chunk_size=chunk_size)
    raw_chunks: list[tuple[int, str]] = []

    for page_num, _, content in pages:
        for chunk_text in chunker.split_text(content):
            if chunk_text.strip():
                raw_chunks.append((page_num, chunk_text.strip()))

    total_chunks = len(raw_chunks)
    return [
        Chunk(
            text=chunk_text,
            file_name=file_name,
            page_number=page_num,
            total_pages=total_pages_global,
            total_chunks=total_chunks,
            chunk_index=idx
        )
        for idx, (page_num, chunk_text) in enumerate(raw_chunks)
    ]


def _split_md_into_pages(raw: str) -> list[tuple[int, int, str]]:
    parts = re.split(r"(={5} Trang \d+ / \d+ ={5})", raw)
    pages: list[tuple[int, int, str]] = []
    i = 1
    while i < len(parts):
        m = _PAGE_SEP_RE.match(parts[i].strip())
        if m:
            page_num, total_pages = int(m.group(1)), int(m.group(2))
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content:
                pages.append((page_num, total_pages, content))
            i += 2
        else:
            i += 1
    return pages


def _chunk_and_classify(
        md_path: str,
        xlsx_path: str,
        chunk_size: int,
        start_step: int,
) -> list[Chunk]:
    logger.info("[%d] Chunking markdown by page…", start_step)
    chunks = chunk_md_file(md_path, chunk_size=chunk_size)
    logger.info("      %d chunks created", len(chunks))

    logger.info("[%d] Classifying nodes…", start_step + 1)
    NodeClassifier(xlsx_path).classify_all(chunks)
    logger.info("      Classification complete")

    return chunks


def run_pipeline_pdf(
        pdf_path: str,
        xlsx_path: str,
        chunk_size: int = 800,
) -> list[Chunk]:
    """PDF → MD → Chunks → Node classification."""
    logger.info("[1/3] Extracting PDF content…")
    md_filename = Path(pdf_path).stem + ".md"
    md_path = PREPARES_DIR / md_filename
    md_content = clean_text(process_pdf(pdf_path))
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("      Saved: %s", md_path)

    return _chunk_and_classify(str(md_path), xlsx_path, chunk_size, start_step=2)


def run_pipeline_md(
        md_path: str,
        xlsx_path: str,
        chunk_size: int = 800,
) -> list[Chunk]:
    """MD → Chunks → Node classification."""
    return _chunk_and_classify(md_path, xlsx_path, chunk_size, start_step=1)
