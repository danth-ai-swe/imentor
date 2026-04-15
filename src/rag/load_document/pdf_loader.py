import sys
from pathlib import Path

import pymupdf
from src.utils.logger_utils import logger


def pdf_to_markdown(pdf_path: Path) -> str:
    """
    Extracts text content from a PDF file and returns it as a Markdown string.
    Each page is separated by a Markdown header.
    """
    try:
        doc = pymupdf.open(str(pdf_path))  # open a document
        md_content = []
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text()  # get plain text encoded as UTF-8
            md_content.append(text.strip())
        return "".join(md_content)
    except Exception as e:
        logger.error(f"❌ Failed to extract PDF {pdf_path}: {e}")
        return ""
