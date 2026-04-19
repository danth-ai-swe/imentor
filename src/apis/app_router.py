from fastapi import APIRouter
documents_router = APIRouter(prefix="/vector/documents", tags=["Vector Documents"])
search_router = APIRouter(prefix="/vector/search", tags=["Vector Search"])
file_router = APIRouter(prefix="/file", tags=["File Search"])
filters_router = APIRouter(prefix="/vector/filters", tags=["Vector Filters"])
chat_router = APIRouter(prefix="/chat", tags=["Chat"])
quiz_router = APIRouter(prefix="/quiz", tags=["Quiz"])

# Import controller so that @router decorators register their routes.
from src.apis import app_controller  # noqa: E402,F401

api_router = APIRouter()
api_router.include_router(documents_router)
api_router.include_router(search_router)
api_router.include_router(filters_router)
api_router.include_router(chat_router)
api_router.include_router(file_router)
api_router.include_router(quiz_router)
