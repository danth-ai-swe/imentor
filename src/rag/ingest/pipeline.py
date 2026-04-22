import json

from qdrant_client import models

from src.constants.app_constant import COLLECTION_NAME
from src.constants.app_constant import (
    INGEST_DIR,
)
from src.rag.db_vector import get_qdrant_client
from src.utils.checkpoint_utils import (
    load_checkpoint as _load_checkpoint_raw,
    save_checkpoint as _save_checkpoint_raw,
    clear_checkpoint,
)
from src.utils.logger_utils import logger, StepTimer


def load_checkpoint() -> set[str]:
    data = _load_checkpoint_raw()
    return set(data["completed"])


def save_checkpoint(completed: set[str]) -> None:
    data = _load_checkpoint_raw()
    data["completed"] = sorted(completed)
    _save_checkpoint_raw(data)


async def upload_to_qdrant(force_restart: bool = False, collection_name: str | None = None):
    target_collection = collection_name or COLLECTION_NAME
    timer = StepTimer("ingest_upload")

    try:
        async with timer.astep("prepare_checkpoint"):
            if force_restart:
                clear_checkpoint()

            uploaded_files = load_checkpoint()
            all_json_files = sorted(INGEST_DIR.glob("*.json"))
            pending_files = [f for f in all_json_files if f.name not in uploaded_files]

        logger.info(f"📊 Tổng file: {len(all_json_files)}")
        logger.info(f"✅ Đã upload:  {len(uploaded_files)}")
        logger.info(f"⏳ Còn lại:    {len(pending_files)}\n")

        if not pending_files:
            logger.info("🎉 Tất cả file đã được upload rồi!")
            return

        async with timer.astep("create_collection_and_indexes"):
            manager = get_qdrant_client()
            manager.collection_name = target_collection
            await manager.acreate_collection(recreate=force_restart)
            await manager.acreate_payload_index("category", models.PayloadSchemaType.KEYWORD)
            await manager.acreate_payload_index("file_name", models.PayloadSchemaType.KEYWORD)

        for i, json_file in enumerate(pending_files, 1):
            logger.info(f"[{i}/{len(pending_files)}] 📂 {json_file.name} ...")

            async with timer.astep(f"upload_file_{json_file.stem}"):
                with open(json_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Lỗi parse JSON: {e}")
                        continue

                if not isinstance(data, list):
                    logger.error(f"❌ Không phải list, bỏ qua.")
                    continue

                try:
                    await manager.aupload_documents(
                        data,
                        batch_size=50,
                        max_retries=3,
                    )
                    uploaded_files.add(json_file.name)
                    save_checkpoint(uploaded_files)
                    logger.info(f"✅ {len(data)} chunks")

                except Exception as e:
                    logger.exception(f"❌ Upload thất bại: {json_file.name}")
                    logger.info("💾 Checkpoint đã lưu tiến độ, chạy lại để tiếp tục.")
                    return

        logger.info(f"\n🚀 Hoàn tất! Đã upload {len(uploaded_files)}/{len(all_json_files)} files.")
    finally:
        timer.summary()
