"""
entrypoint.py — PDF / MD → chunk → classify pipeline
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import fitz
import pandas as pd

from src.constants.app_constant import PREPARES_DIR
from src.rag.chunking.recursive_chunker import get_recursive_token_chunk
from src.rag.clean_data.prompt import (
    CLASSIFY_NODE_PROMPT,
    CLASSIFY_PROMPT,
    EXTRACT_TABLE_PROMPT,
    EXTRACT_CHART_PROMPT,
    EXTRACT_TEXT_PROMPT,
)
from src.rag.llm.chat_llm import get_openai_chat_client
from src.utils.app_utils import clean_text

logger = logging.getLogger(__name__)

MIN_IMAGE_PIXELS = 5_000
_PAGE_SEP_RE = re.compile(r"={5} Trang (\d+) / (\d+) ={5}")
_FILENAME_RE = re.compile(r"^(LOMA\d+)_M(\d+)L(\d+)")

_LABEL_MAP = {
    "table": "**[TABLE]**",
    "chart": "**[CHART]**",
    "text_data": "**[TEXT DATA]**",
}
_EXTRACT_PROMPT_MAP = {
    "table": EXTRACT_TABLE_PROMPT,
    "chart": EXTRACT_CHART_PROMPT,
    "text_data": EXTRACT_TEXT_PROMPT,
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    file_name: str
    page_number: int
    total_pages: int
    total_chunks: int
    chunk_index: int
    course: Optional[str] = None
    module: Optional[str] = None
    lesson: Optional[str] = None
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    category: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_filename_meta(file_name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Trích xuất course / module / lesson từ tên file."""
    m = _FILENAME_RE.search(file_name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None


# ── Image processing ──────────────────────────────────────────────────────────

class ImageProcessor:
    """Phân loại và trích xuất nội dung từ ảnh nhúng trong PDF."""

    def __init__(self, client) -> None:
        self.client = client

    def pixmap_to_base64(self, pix: fitz.Pixmap) -> str:
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")

    def classify(self, b64: str) -> dict:
        raw = self.client.invoke_with_image(
            prompt=CLASSIFY_PROMPT,
            image_base64=b64,
            media_type="image/png",
        )
        raw = _strip_json_fence(raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"type": "other", "reason": raw}

    def extract(self, b64: str, img_type: str) -> str:
        prompt = _EXTRACT_PROMPT_MAP.get(img_type)
        if not prompt:
            return ""
        return self.client.invoke_with_image(
            prompt=prompt,
            image_base64=b64,
            media_type="image/png",
        )

    def process_image(self, doc: fitz.Document, xref: int) -> Optional[str]:
        """
        Xử lý một ảnh: phân loại rồi trích xuất nội dung.
        Trả về chuỗi markdown hoặc None nếu bỏ qua.
        """
        try:
            base_image = doc.extract_image(xref)
        except Exception as exc:
            logger.warning("Không trích xuất được ảnh xref=%s: %s", xref, exc)
            return None

        w, h = base_image.get("width", 0), base_image.get("height", 0)
        if w * h < MIN_IMAGE_PIXELS:
            logger.debug("Bỏ qua ảnh nhỏ (%dx%dpx)", w, h)
            return None

        logger.debug("Xử lý ảnh xref=%s (%dx%dpx)…", xref, w, h)
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha > 3:  # normalize CMYK → RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)

        b64 = self.pixmap_to_base64(pix)
        classification = self.classify(b64)
        img_type = classification.get("type", "other")
        logger.debug("→ Loại: %s | %s", img_type, classification.get("reason", ""))

        if img_type == "other":
            return None

        content = self.extract(b64, img_type)
        label = _LABEL_MAP[img_type]
        return f"{label}\n\n{content}"


# ── PDF extractor ─────────────────────────────────────────────────────────────

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


# ── MD chunker ────────────────────────────────────────────────────────────────

def chunk_md_file(md_path: str, chunk_size: int = 800) -> list[Chunk]:
    """Markdown (phân trang) → list[Chunk]."""
    md_path_obj = Path(md_path)
    file_name = md_path_obj.name
    raw = md_path_obj.read_text(encoding="utf-8")

    pages = _split_md_into_pages(raw, md_path)
    if not pages:
        raise ValueError(f"Không tìm thấy separator trang trong file: {md_path}")

    total_pages_global = pages[0][1]
    course, module, lesson = _parse_filename_meta(file_name)

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
            chunk_index=idx,
            course=course,
            module=module,
            lesson=lesson,
        )
        for idx, (page_num, chunk_text) in enumerate(raw_chunks)
    ]


def _split_md_into_pages(raw: str, md_path: str) -> list[tuple[int, int, str]]:
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


# ── Node classifier ───────────────────────────────────────────────────────────

class NodeClassifier:
    """Phân loại chunk vào node theo metadata_node.xlsx."""

    def __init__(self, xlsx_path: str) -> None:
        self.xlsx_path = xlsx_path
        self.client = get_openai_chat_client()
        self._cache: dict[str, pd.DataFrame] = {}

    def _get_candidates(self, source_prefix: str) -> pd.DataFrame:
        if source_prefix not in self._cache:
            df = pd.read_excel(self.xlsx_path)
            mask = df["Source"].astype(str).str.startswith(source_prefix)
            self._cache[source_prefix] = (
                df[mask][["Node ID", "Node Name", "Category"]].drop_duplicates()
            )
        return self._cache[source_prefix]

    def classify(self, chunk: Chunk) -> None:
        """Gán node_id / node_name / category trực tiếp vào chunk (in-place)."""
        parts = chunk.file_name.split("_")
        source_prefix = "_".join(parts[:2])

        candidates_df = self._get_candidates(source_prefix)
        if candidates_df.empty:
            return  # fields giữ None mặc định

        candidates_str = "\n".join(
            f"{row['Node ID']} | {row['Category']} | {row['Node Name']}"
            for _, row in candidates_df.iterrows()
        )
        prompt = CLASSIFY_NODE_PROMPT.format(
            candidates=candidates_str,
            chunk_text=chunk.text[:2000],
        )
        raw = _strip_json_fence(self.client.invoke(prompt))
        try:
            result = json.loads(raw)
            chunk.node_id = result.get("node_id")
            chunk.node_name = result.get("node_name")
            chunk.category = result.get("category")
        except json.JSONDecodeError:
            logger.warning("Không parse được JSON từ classifier cho chunk %d", chunk.chunk_index)

    def classify_all(self, chunks: list[Chunk]) -> None:
        """Phân loại toàn bộ danh sách chunk (in-place)."""
        total = len(chunks)
        for chunk in chunks:
            logger.info(
                "  Classifying chunk %d/%d (page %d)…",
                chunk.chunk_index + 1, total, chunk.page_number,
            )
            self.classify(chunk)


# ── Pipeline functions ────────────────────────────────────────────────────────

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


# ── Utility ───────────────────────────────────────────────────────────────────

def _strip_json_fence(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()