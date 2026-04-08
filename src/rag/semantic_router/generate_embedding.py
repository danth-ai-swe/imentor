from pathlib import Path

import numpy as np

from src.rag.llm.embedding_llm import get_openai_embedding_client
from src.rag.semantic_router.samples import (
    coreKnowledgeSamples,
    offTopicSamples,
)
from src.utils.logger_utils import logger, log_function_call

OUTPUT_PATH = Path(__file__).with_name("route_embeddings.npz")

ROUTES = {
    "core_knowledge": coreKnowledgeSamples,
    "off_topic": offTopicSamples,
}


@log_function_call
def generate() -> None:
    embedder = get_openai_embedding_client()
    arrays: dict[str, np.ndarray] = {}

    for route_name, samples in ROUTES.items():
        logger.info(f"⏳ Embedding route '{route_name}' — {len(samples)} samples …")
        vectors = embedder.embed_documents(samples)
        arrays[route_name] = np.array(vectors, dtype=np.float32)
        logger.info(f"✅ '{route_name}' shape={arrays[route_name].shape}")

    np.savez(OUTPUT_PATH, **arrays)
    logger.info(f"💾 Saved pre-computed embeddings → {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
