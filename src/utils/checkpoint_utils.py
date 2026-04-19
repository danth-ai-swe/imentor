import json
from datetime import datetime
from pathlib import Path

from src.utils.logger_utils import logger

CHECKPOINT_FILE = Path("checkpoint.json")


def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
            data.setdefault("completed", [])
            data.setdefault("failed", {})
            return data
        except Exception as e:
            logger.warning(f"⚠️  Could not read checkpoint: {e} — starting fresh.")
    return {"completed": [], "failed": {}}


def save_checkpoint(checkpoint: dict) -> None:
    try:
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_FILE.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.error(f"❌ Failed to save checkpoint: {e}")


def mark_completed(checkpoint: dict, filename: str) -> None:
    if filename not in checkpoint["completed"]:
        checkpoint["completed"].append(filename)
    checkpoint["failed"].pop(filename, None)
    save_checkpoint(checkpoint)


def mark_failed(checkpoint: dict, filename: str, error: str) -> None:
    checkpoint["failed"][filename] = {
        "error": error,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_checkpoint(checkpoint)


def clear_checkpoint() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("🗑️  Checkpoint cleared.")
