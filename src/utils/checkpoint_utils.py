import json
from datetime import datetime
from pathlib import Path

from src.utils.logger_utils import logger

CHECKPOINT_FILE = Path("checkpoint.json")


def load_checkpoint() -> dict:
    """
    Đọc checkpoint.json. Trả về dict:
      { "completed": [...], "failed": {...} }
    Nếu file chưa tồn tại, trả về cấu trúc rỗng.
    """
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
    """Ghi checkpoint dict xuống file (overwrite)."""
    try:
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_FILE.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.error(f"❌ Failed to save checkpoint: {e}")


def mark_completed(checkpoint: dict, filename: str) -> None:
    """Đánh dấu file đã xử lý thành công."""
    if filename not in checkpoint["completed"]:
        checkpoint["completed"].append(filename)
    # Xoá khỏi failed nếu trước đó bị lỗi và chạy lại thành công
    checkpoint["failed"].pop(filename, None)
    save_checkpoint(checkpoint)


def mark_failed(checkpoint: dict, filename: str, error: str) -> None:
    """Ghi nhận file bị lỗi kèm thông báo lỗi."""
    checkpoint["failed"][filename] = {
        "error": error,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_checkpoint(checkpoint)


def clear_checkpoint() -> None:
    """Xoá file checkpoint để bắt đầu lại từ đầu."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("🗑️  Checkpoint cleared.")


def get_completed_set(checkpoint: dict) -> set[str]:
    """Trả về set tên file đã completed (tiện cho tra cứu nhanh)."""
    return set(checkpoint["completed"])
