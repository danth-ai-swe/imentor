from pathlib import Path

from src.utils.logger_utils import logger


def load_markdown_file(md_path: Path) -> str:
    try:
        return md_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"❌ Failed to load {md_path}: {e}")
        return ""
