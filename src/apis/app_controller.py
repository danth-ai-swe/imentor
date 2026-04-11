import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import BackgroundTasks, Depends
from fastapi.responses import FileResponse

from src.apis.app_exception import BadRequestError, InvalidFilterError, NotFoundError, QdrantApiError
from src.apis.app_model import (
    ChatRequest,
    ChatResponse,
    ChatDataModel,
    FilterConditionModel,
    GetByIdsRequest,
    HybridSearchRequest,
    GenerateEmbeddingsResponse, IntentRouteResponse, IntentRouteRequest, UploadDocumentsRequest,
    BatchSearchRequest, CreatePayloadIndexRequest, QuizRequest
)
from src.apis.app_router import chat_router, documents_router, search_router, file_router, quiz_router
from src.constants.app_constant import PDFS_DIR, COLLECTION_NAME, DATA_ZIP, SRC_ZIP
from src.core.quiz.quiz_generator import generate_quiz
from src.rag.db_vector import get_qdrant_client
from src.rag.ingest.entrypoint import upload_to_qdrant
from src.rag.llm.embedding_llm import get_openai_embedding_client
from src.rag.search.pipeline import async_pipeline_hyde_search
from src.rag.semantic_router.generate_embeddings import ROUTES, OUTPUT_PATH, generate
from src.rag.semantic_router.router import load_precomputed_embeddings, Route, SemanticRouter
from src.utils.logger_utils import logger


def _to_qdrant_conditions(
        conditions: list[FilterConditionModel] | None,
) -> list[Any] | None:
    if not conditions:
        return None

    try:
        from qdrant_client.http import models

        return [
            models.FieldCondition(
                key=condition.key,
                match=models.MatchValue(value=condition.value),
            )
            for condition in conditions
        ]
    except Exception as exc:
        logger.exception("Invalid filter conditions")
        raise InvalidFilterError(str(exc)) from exc


def _to_plain(data: Any) -> Any:
    if data is None or isinstance(data, (str, int, float, bool)):
        return data
    if isinstance(data, list):
        return [_to_plain(item) for item in data]
    if isinstance(data, tuple):
        return tuple(_to_plain(item) for item in data)
    if isinstance(data, dict):
        return {key: _to_plain(value) for key, value in data.items()}
    if hasattr(data, "model_dump"):
        return _to_plain(data.model_dump())
    if hasattr(data, "dict"):
        return _to_plain(data.dict())
    if hasattr(data, "__dict__"):
        return _to_plain(vars(data))
    return str(data)


@chat_router.post("/intent-route", response_model=IntentRouteResponse)
async def intent_route(payload: IntentRouteRequest) -> IntentRouteResponse:
    texts = [payload.texts] if isinstance(payload.texts, str) else payload.texts
    if not texts:
        raise BadRequestError("texts must not be empty")

    try:
        embedder = get_openai_embedding_client()

        precomputed = load_precomputed_embeddings()
        if precomputed is None:
            raise NotFoundError(resource="route_embeddings.npz", id="route_embeddings.npz")

        routes = [Route(name=k) for k in precomputed.keys()]

        router = SemanticRouter(
            embedding=embedder,
            routes=routes,
            precomputed_embeddings=precomputed,
        )

        results = []
        for text in texts:
            score, intent = await router.aguide(text)
            results.append({
                "text": text,
                "intent": intent,
                "score": float(round(score, 6)),
            })

        return IntentRouteResponse(results=results)

    except NotFoundError:
        raise
    except Exception as exc:
        logger.exception("intent_route failed")
        raise QdrantApiError(f"intent_route failed: {exc}") from exc


@documents_router.get("/ingest")
async def ingest_documents(force_restart: bool = False, collection_name: str | None = None,
                           background_tasks: BackgroundTasks = None):
    target_collection = collection_name or COLLECTION_NAME
    try:
        if not force_restart:
            manager = get_qdrant_client()
            manager.collection_name = target_collection
            exists = await manager.async_client.collection_exists(target_collection)
            if exists:
                info = await manager.async_client.get_collection(target_collection)
                if info.points_count and info.points_count > 0:
                    return {
                        "status": "skipped",
                        "message": f"Collection '{target_collection}' already exists with {info.points_count} points. Use force_restart=true to re-ingest.",
                        "points_count": info.points_count,
                    }
    except Exception as exc:
        logger.exception("Ingest failed")
        raise QdrantApiError(f"Ingest failed: {exc}") from exc

    background_tasks.add_task(upload_to_qdrant, force_restart=force_restart, collection_name=target_collection)
    return {"status": "started", "message": f"Ingest triggered in background for collection '{target_collection}'."}


@documents_router.post("/by-ids")
async def get_by_ids(payload: GetByIdsRequest, manager=Depends(get_qdrant_client)) -> dict[str, Any]:
    try:
        fetched = await manager.aget_by_ids(
            payload.ids,
            with_vectors=payload.with_vectors,
            with_payload=payload.with_payload,
        )
        return {"items": _to_plain(fetched), "count": len(fetched)}
    except Exception as exc:
        logger.exception("get_by_ids failed")
        raise QdrantApiError(f"get_by_ids failed: {exc}") from exc


@search_router.post("/hybrid")
async def hybrid_search(payload: HybridSearchRequest, manager=Depends(get_qdrant_client)) -> dict[str, Any]:
    try:
        results = await manager.ahybrid_search(
            query=payload.query,
            top_k=payload.top_k,
            prefetch_limit=payload.prefetch_limit,
            filter_conditions=_to_qdrant_conditions(payload.filter_conditions),
            with_payload=payload.with_payload,
        )
        return {"items": _to_plain(results), "count": len(results)}
    except Exception as exc:
        logger.exception("hybrid_search failed")
        raise QdrantApiError(f"hybrid_search failed: {exc}") from exc


@chat_router.post("/ask", response_model=ChatResponse)
async def chat_ask(payload: ChatRequest) -> ChatResponse:
    logger.info("chat_ask called | user_name=%s | conversation_id=%s", payload.user_name, payload.conversation_id)
    try:
        result = await async_pipeline_hyde_search(
            user_input=payload.message,
            conversation_id=payload.conversation_id
        )
    except Exception as exc:
        logger.exception("pipeline_search failed")
        raise QdrantApiError(f"pipeline_search failed: {exc}") from exc

    answer_satisfied = result.get("answer_satisfied", True)
    content = result.get("response", "") or ""
    intent = result.get("intent")
    sources = result.get("sources")
    if not answer_satisfied:
        sources = []
    return ChatResponse(
        success=True,
        data=ChatDataModel(
            role="assistant",
            intent=intent,
            content=content,
            sources=sources,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )


@chat_router.post("/generate-embeddings", response_model=GenerateEmbeddingsResponse)
async def generate_route_embeddings() -> GenerateEmbeddingsResponse:
    try:

        await asyncio.to_thread(generate)

        data = np.load(OUTPUT_PATH)
        routes_info = [
            {
                "route": k,
                "samples": ROUTES[k].__len__() if hasattr(ROUTES[k], "__len__") else -1,
                "shape": list(data[k].shape),
            }
            for k in data.files
        ]

        return GenerateEmbeddingsResponse(
            routes=routes_info,
            output_path=str(OUTPUT_PATH),
        )

    except Exception as exc:
        logger.exception("generate_embeddings failed")
        raise QdrantApiError(f"generate_embeddings failed: {exc}") from exc


@documents_router.post("/collections/create")
async def create_collection(recreate: bool = False, manager=Depends(get_qdrant_client)) -> dict:
    try:
        await manager.acreate_collection(recreate=recreate)
        return {"created": True, "collection": manager.collection_name}
    except Exception as exc:
        logger.exception("create_collection failed")
        raise QdrantApiError(f"create_collection failed: {exc}") from exc


@documents_router.delete("/collections")
async def delete_collection(manager=Depends(get_qdrant_client)) -> dict:
    try:
        await manager.adelete_collection()
        return {"deleted": True, "collection": manager.collection_name}
    except Exception as exc:
        logger.exception("delete_collection failed")
        raise QdrantApiError(f"delete_collection failed: {exc}") from exc


@documents_router.post("/upload")
async def upload_documents(payload: UploadDocumentsRequest, manager=Depends(get_qdrant_client)) -> dict:
    try:
        await manager.aupload_documents(
            documents=payload.documents,
            batch_size=payload.batch_size,
            parallel=payload.parallel,
            max_retries=payload.max_retries,
        )
        return {"uploaded": len(payload.documents)}
    except Exception as exc:
        logger.exception("upload_documents failed")
        raise QdrantApiError(f"upload_documents failed: {exc}") from exc


@documents_router.post("/indexes/payload")
async def create_payload_index(payload: CreatePayloadIndexRequest, manager=Depends(get_qdrant_client)) -> dict:
    try:
        from qdrant_client.http import models as qmodels
        schema_map = {
            "keyword": qmodels.PayloadSchemaType.KEYWORD,
            "integer": qmodels.PayloadSchemaType.INTEGER,
            "float": qmodels.PayloadSchemaType.FLOAT,
            "bool": qmodels.PayloadSchemaType.BOOL,
            "text": qmodels.PayloadSchemaType.TEXT,
        }
        schema = schema_map.get(payload.field_schema.lower())
        if schema is None:
            raise BadRequestError(f"Invalid field_schema '{payload.field_schema}'")
        await manager.acreate_payload_index(payload.field_name, schema)
        return {"indexed": True, "field": payload.field_name}
    except BadRequestError:
        raise
    except Exception as exc:
        logger.exception("create_payload_index failed")
        raise QdrantApiError(f"create_payload_index failed: {exc}") from exc


@search_router.post("/batch")
async def batch_search(payload: BatchSearchRequest, manager=Depends(get_qdrant_client)) -> dict:
    try:
        import asyncio
        tasks = [
            manager.ahybrid_search(q, top_k=payload.top_k, prefetch_limit=payload.prefetch_limit)
            for q in payload.queries
        ]
        results = await asyncio.gather(*tasks)
        return {
            "batches": [
                {"query": q, "items": _to_plain(r), "count": len(r)}
                for q, r in zip(payload.queries, results)
            ]
        }
    except Exception as exc:
        logger.exception("batch_search failed")
        raise QdrantApiError(f"batch_search failed: {exc}") from exc


@file_router.get("/data")
def get_ingest_zip_file():
    if not os.path.isfile(DATA_ZIP):
        raise NotFoundError(resource="DATA_ZIP file not found", id=DATA_ZIP)
    return FileResponse(DATA_ZIP, filename=os.path.basename(DATA_ZIP), media_type="application/zip")


@file_router.get("/src")
def get_ingest_zip_file():
    if not os.path.isfile(SRC_ZIP):
        raise NotFoundError(resource="SRC_ZIP file not found", id=SRC_ZIP)
    return FileResponse(SRC_ZIP, filename=os.path.basename(SRC_ZIP), media_type="application/zip")


@file_router.get("/{filename:path}")
def get_pdf_file(filename: str):
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise BadRequestError("Invalid file_name path")

    pdf_file = filename if filename.lower().endswith(".pdf") else filename + ".pdf"
    file_path = os.path.join(PDFS_DIR, pdf_file)

    if not os.path.isfile(file_path):
        raise NotFoundError(resource="file name not found", id=filename)

    response = FileResponse(
        file_path,
        media_type="application/pdf",
        filename=pdf_file
    )
    response.headers["Content-Disposition"] = f'inline; filename="{pdf_file}"'
    return response


@quiz_router.post("/generate")
def generate(payload: QuizRequest) -> dict:
    try:
        return generate_quiz(
            knowledge_pack=payload.knowledge_pack,
            total=payload.total,
            difficulty=payload.difficulty.value if payload.difficulty else None,
            module_value=payload.module_value,
            lesson_value=payload.lesson_value,
        )
    except ValueError as exc:
        logger.exception("generate_quiz validation error")
        raise BadRequestError(str(exc)) from exc
    except Exception as exc:
        logger.exception("generate_quiz failed")
        raise QdrantApiError(f"generate_quiz failed: {exc}") from exc
