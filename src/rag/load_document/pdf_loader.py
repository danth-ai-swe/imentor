import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pymupdf  # imports the pymupdf library
import pdfplumber

from src.utils.logger_utils import logger
from src.utils.app_utils import clean_text, clean_text_to_one_line


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


def extract_text_with_pdfplumber(pdf_path: Path) -> str:
    """
    Extracts all text from a PDF file using pdfplumber and returns it as a single string.
    Each page's text is separated by a page header.
    """
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            all_text = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                all_text.append(f"\n\n# Page {i + 1}\n\n{text.strip()}")
            return "".join(all_text)
    except Exception as e:
        logger.error(f"❌ Failed to extract PDF with pdfplumber {pdf_path}: {e}")
        return ""


if __name__ == '__main__':
    # Example usage
    pdf_path = Path(r"C:\imt-ai-brain\data\summary\LOMA291_M2L2_Summary.pdf")
    markdown = pdf_to_markdown(pdf_path)
    print(clean_text_to_one_line(markdown))
