from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

# PDF
import fitz  # pymupdf

from src.rag.load_document.base_document import BaseDocumentExtractor
from src.utils.logger_utils import log


class PDFExtractor(BaseDocumentExtractor):
    def _extract_blocks(self, file_path: Path) -> List[Tuple[int, str, bytes | str]]:
        doc = fitz.open(str(file_path))
        total_pages = len(doc)

        all_blocks: List[Tuple[int, str, bytes | str]] = []

        def _process_page(page_num: int) -> List[Tuple[int, str, bytes | str]]:
            page = doc[page_num]
            page_blocks: List[Tuple[int, str, bytes | str]] = []
            base_index = page_num * 100_000  # ample space per page

            # --- Text blocks ---
            text_dict = page.get_text("dict", sort=True)  # type: ignore[arg-type]
            text_order = 0
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # text
                    lines = []
                    for line in block.get("lines", []):
                        spans = [s.get("text", "") for s in line.get("spans", [])]
                        lines.append("".join(spans))
                    block_text = "\n".join(lines)
                    if block_text.strip():
                        page_blocks.append(
                            (base_index + text_order, "text", block_text)
                        )
                        text_order += 1

            # --- Image blocks ---
            img_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(img_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    # Use a high offset so images come after text of same page
                    img_order = base_index + 50_000 + img_idx
                    page_blocks.append((img_order, "image", img_bytes))
                except Exception as exc:
                    log.warning("PDF page %d image %d error: %s", page_num, img_idx, exc)

            return page_blocks

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_process_page, p): p for p in range(total_pages)}
            for future in as_completed(futures):
                try:
                    all_blocks.extend(future.result())
                except Exception as exc:
                    log.warning("PDF page processing error: %s", exc)

        doc.close()
        return all_blocks
