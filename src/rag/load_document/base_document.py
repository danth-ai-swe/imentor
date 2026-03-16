from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from src.rag.llm.vision_llm import get_vision_client
from src.utils.logger_utils import log


class BaseDocumentExtractor(ABC):
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._vision = get_vision_client()

    def extract(self, file_path: str | Path) -> str:
        """
        Main entry point.
        Returns the full text content of the document (text + image descriptions).
        """
        file_path = Path(file_path)
        blocks = self._extract_blocks(file_path)  # [(idx, kind, content), ...]
        blocks = self._convert_images_parallel(blocks)  # images → text in parallel
        return self._assemble(blocks)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def _extract_blocks(
            self, file_path: Path
    ) -> List[Tuple[int, str, bytes | str]]:
        """
        Extract ordered content blocks from the document.

        Returns a list of tuples:
            (order_index, kind, content)
            kind  = "text"  → content is str
            kind  = "image" → content is bytes (raw image data)
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_images_parallel(
            self, blocks: List[Tuple[int, str, bytes | str]]
    ) -> List[Tuple[int, str, str]]:
        """Replace image blocks with their text description (parallel)."""
        image_indices = [i for i, (_, kind, _) in enumerate(blocks) if kind == "image"]

        if not image_indices:
            return blocks  # type: ignore[return-value]

        result: Dict[int, str] = {}

        def _process(pos: int) -> Tuple[int, str]:
            _, _, img_bytes = blocks[pos]
            text = self._vision.image_to_text(img_bytes)  # type: ignore[arg-type]
            return pos, text

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_process, pos): pos for pos in image_indices}
            for future in as_completed(futures):
                try:
                    pos, text = future.result()
                    result[pos] = text
                except Exception as exc:
                    pos = futures[future]
                    log.warning("Image-to-text failed at block %d: %s", pos, exc)
                    result[pos] = "[Ảnh: không thể trích xuất nội dung]"

        # Merge back
        converted: List[Tuple[int, str, str]] = []
        for pos, (idx, kind, content) in enumerate(blocks):
            if kind == "image":
                converted.append((idx, "image_text", result.get(pos, "")))
            else:
                converted.append((idx, kind, content))  # type: ignore[arg-type]
        return converted

    @staticmethod
    def _assemble(blocks: List[Tuple[int, str, str]]) -> str:
        """Concatenate all blocks in order into a single string."""
        parts: List[str] = []
        for _, kind, content in sorted(blocks, key=lambda x: x[0]):
            if not content or not content.strip():
                continue
            if kind == "image_text":
                parts.append(f"[Nội dung ảnh: {content.strip()}]")
            else:
                parts.append(content.strip())
        return "\n\n".join(parts)
