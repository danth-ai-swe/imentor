from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from src.rag.load_document.base_document import BaseDocumentExtractor
from src.rag.load_document.docx_type import DOCXExtractor
from src.rag.load_document.pdf_type import PDFExtractor
from src.rag.load_document.pptx_type import PPTXExtractor
from src.rag.load_document.xlsx_type import XLSXExtractor
from src.utils.logger_utils import log

_EXTRACTOR_MAP: Dict[str, type] = {
    ".pdf": PDFExtractor,
    ".docx": DOCXExtractor,
    ".doc": DOCXExtractor,
    ".pptx": PPTXExtractor,
    ".ppt": PPTXExtractor,
    ".xlsx": XLSXExtractor,
    ".xls": XLSXExtractor,
}


def extract_document(
        file_path: str | Path,
        max_workers: int = 4,
) -> str:
    """
    Convenience function: auto-detect file type and extract full text content.

    Args:
        file_path: Đường dẫn tới file cần trích xuất.
        max_workers: Số luồng song song tối đa.

    Returns:
        Toàn bộ nội dung text của file (kể cả text từ ảnh).
    """
    ext = Path(file_path).suffix.lower()
    extractor_cls = _EXTRACTOR_MAP.get(ext)
    if extractor_cls is None:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {list(_EXTRACTOR_MAP.keys())}"
        )
    extractor: BaseDocumentExtractor = extractor_cls(max_workers=max_workers)
    return extractor.extract(file_path)


def extract_documents_parallel(
        file_paths: List[str | Path],
        max_workers: int = 4,
) -> Dict[str, str]:
    """
    Trích xuất nhiều file song song.

    Args:
        file_paths: Danh sách đường dẫn file.
        max_workers: Số luồng song song.

    Returns:
        Dict { file_path_str: extracted_text }
    """
    results: Dict[str, str] = {}

    def _extract(fp: str | Path) -> Tuple[str, str]:
        text = extract_document(fp, max_workers=max_workers)
        return str(fp), text

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_extract, fp): fp for fp in file_paths}
        for future in as_completed(futures):
            fp = futures[future]
            try:
                path_str, text = future.result()
                results[path_str] = text
            except Exception as exc:
                log.error("Failed to extract %s: %s", fp, exc)
                results[str(fp)] = f"[Lỗi trích xuất: {exc}]"

    return results


if __name__ == '__main__':
    # Một file
    text = extract_document("../../../data/test.pdf")
    print(text)

    # Nhiều file song song
    results = extract_documents_parallel(
        ["slide.pptx", "data.xlsx", "contract.docx", "scan.pdf"],
        max_workers=4,
    )
