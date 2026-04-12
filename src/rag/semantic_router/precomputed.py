import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.rag.search.pipeline import INTENT_CORE_KNOWLEDGE, INTENT_OFF_TOPIC
from src.rag.semantic_router.samples import offTopicSamples, coreKnowledgeSamples

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache/embeddings")
CACHE_FILE = CACHE_DIR / "intent_routes.npz"
CHECKSUM_FILE = CACHE_DIR / "intent_routes.checksum"

ROUTE_SAMPLES: Dict[str, list] = {
    INTENT_CORE_KNOWLEDGE: coreKnowledgeSamples,
    INTENT_OFF_TOPIC: offTopicSamples,
}


def _compute_checksum() -> str:
    """Hash toàn bộ samples — nếu samples thay đổi thì cache bị invalidate."""
    payload = json.dumps(ROUTE_SAMPLES, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def _is_cache_valid() -> bool:
    if not CACHE_FILE.exists() or not CHECKSUM_FILE.exists():
        return False
    stored = CHECKSUM_FILE.read_text().strip()
    return stored == _compute_checksum()


def load_precomputed_embeddings() -> Optional[Dict[str, np.ndarray]]:
    """
    Load embeddings từ file cache nếu còn hợp lệ.
    Trả về None nếu cache chưa có hoặc samples đã thay đổi
    → SemanticRouter.abuild() sẽ tự encode lại.
    """
    if not _is_cache_valid():
        logger.info("Precomputed embeddings cache miss hoặc outdated.")
        return None

    data = np.load(CACHE_FILE)
    embeddings = {name: data[name] for name in data.files}
    logger.info(f"Loaded precomputed embeddings: {list(embeddings.keys())}")
    return embeddings

async def build_and_save_embeddings(embedder) -> Dict[str, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    embeddings: Dict[str, np.ndarray] = {}

    for route_name, samples in ROUTE_SAMPLES.items():
        logger.info(f"Encoding {len(samples)} samples cho route '{route_name}'...")
        vecs = await embedder.aembed_documents(samples)  # List[List[float]]
        embeddings[route_name] = np.array(vecs)          # → shape: (N, dim)

    np.savez_compressed(CACHE_FILE, **embeddings)
    CHECKSUM_FILE.write_text(_compute_checksum())
    logger.info(f"Saved precomputed embeddings → {CACHE_FILE}")
    return embeddings