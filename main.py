from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI

from src.apis.app_router import api_router
from src.config.app_config import get_app_config

config = get_app_config()
from src.rag.db_vector import get_qdrant_client
from src.rag.llm.chat_llm import get_openai_chat_client
from src.rag.llm.embedding_llm import get_openai_embedding_client
from src.utils.logger_utils import setup_uvicorn_logging, get_uvicorn_log_config, RequestTimingMiddleware


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Warm-up Qdrant
    qdrant = get_qdrant_client()
    try:
        qdrant.client.get_collections()
    except Exception:
        pass

    get_openai_chat_client()
    get_openai_embedding_client()

    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(RequestTimingMiddleware)
app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["Health Check"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "git_commit_id": config.GIT_COMMIT_ID,
        "definition_name": config.DEFINITION_NAME,
        "app_build_number": config.APP_BUILD_NUMBER
    }


if __name__ == '__main__':
    setup_uvicorn_logging()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        timeout_keep_alive=1800,
        log_config=get_uvicorn_log_config(),
    )
