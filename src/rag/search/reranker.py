import asyncio
import threading
from typing import List

from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.constants.app_constant import FASTEMBED_CACHE_DIR
from src.rag.search.model import ChunkDict
from src.utils.logger_utils import logger

_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-12-v2"

_reranker_singleton: TextCrossEncoder | None = None
_reranker_lock = threading.Lock()


def _get_reranker() -> TextCrossEncoder:
    global _reranker_singleton
    if _reranker_singleton is None:
        with _reranker_lock:
            if _reranker_singleton is None:
                _reranker_singleton = TextCrossEncoder(
                    _RERANKER_MODEL,
                    cache_dir=str(FASTEMBED_CACHE_DIR),
                )
    return _reranker_singleton


def _rerank_sync(query: str, texts: List[str]) -> List[float]:
    return list(_get_reranker().rerank(query, texts))


async def arerank_chunks(
        query: str,
        chunks: List[ChunkDict],
        top_k: int,
) -> List[ChunkDict]:
    if not chunks or top_k <= 0:
        return []

    try:
        texts = [c.get("text", "") for c in chunks]
        scores = await asyncio.to_thread(_rerank_sync, query, texts)
    except Exception:
        logger.exception("Reranker failed — falling back to vector order")
        return chunks[:top_k]

    for chunk, score in zip(chunks, scores):
        chunk["score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
    return ranked[:top_k]
