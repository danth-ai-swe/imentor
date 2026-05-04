import os
from contextlib import asynccontextmanager
from datetime import datetime

from src.constants.app_constant import FASTEMBED_CACHE_DIR, INTENT_OVERALL_COURSE_KNOWLEDGE

# Route fastembed downloads (incl. those triggered indirectly via qdrant-client)
# to a project-local folder. Must be set before any fastembed model is constructed.
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(FASTEMBED_CACHE_DIR))

import uvicorn
from fastapi import FastAPI

from src.apis.app_router import api_router
from src.config.app_config import get_app_config
from src.rag.db_vector import get_qdrant_client
from src.rag.llm.chat_llm import get_openai_chat_client
from src.rag.llm.embedding_llm import (
    get_openai_embedding_client,
    get_sync_client,
    get_async_client,
)
from src.rag.search.pipeline import INTENT_CORE_KNOWLEDGE, INTENT_OFF_TOPIC
from src.rag.search.reranker import _get_reranker
from src.rag.semantic_router.intent_router_registry import set_intent_router
from src.rag.semantic_router.precomputed import (
    build_and_save_embeddings,
    load_precomputed_embeddings,
)
from src.rag.semantic_router.router import Route, SemanticRouter
from src.rag.semantic_router.samples import offTopicSamples, coreKnowledgeSamples, courseMetadataSamples

config = get_app_config()


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Warm-up Qdrant
    qdrant = get_qdrant_client()
    try:
        qdrant.client.get_collections()
    except Exception:
        pass

    get_sync_client()
    get_async_client()
    get_openai_chat_client()
    embedder = get_openai_embedding_client()
    _get_reranker()
    precomputed = load_precomputed_embeddings()
    if precomputed is None:
        precomputed = await build_and_save_embeddings(embedder)

    router = await SemanticRouter.abuild(
        routes=[
            Route(name=INTENT_CORE_KNOWLEDGE, samples=coreKnowledgeSamples),
            Route(name=INTENT_OFF_TOPIC, samples=offTopicSamples),
            Route(name=INTENT_OVERALL_COURSE_KNOWLEDGE, samples=courseMetadataSamples),
        ],
        embedder=embedder,
        precomputed_embeddings=precomputed,
    )
    set_intent_router(router)

    yield  # ← app đang chạy


app = FastAPI(lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["Health Check"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "git_commit_id": config.GIT_COMMIT_ID,
        "definition_name": config.DEFINITION_NAME,
        "app_build_number": config.APP_BUILD_NUMBER,
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8084,
        timeout_keep_alive=1800,
    )
