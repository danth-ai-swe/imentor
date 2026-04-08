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
    ChatSourceModel,
    DeleteByFilterRequest,
    FilterConditionModel,
    GetByIdsRequest,
    HybridSearchRequest,
    QueryPointsGroupsRequest,
    ScrollRequest, GenerateEmbeddingsResponse, IntentRouteResponse, IntentRouteRequest, UploadDocumentsRequest,
    FacetRequest, BatchSearchRequest, CreatePayloadIndexRequest,
    QuizRequest,
)
from src.apis.app_router import chat_router, documents_router, search_router, file_router, quiz_router
from src.config.app_config import get_app_config
from src.constants.app_constant import PDFS_DIR, COLLECTION_NAME
from src.rag.db_vector import get_qdrant_client
from src.rag.ingest.pipeline import upload_to_qdrant
from src.rag.llm.embedding_llm import get_openai_embedding_client
from src.rag.search.pipeline import async_pipeline_hyde_search
from src.rag.semantic_router.generate_embedding import OUTPUT_PATH, ROUTES, generate
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


@documents_router.post("/scroll")
async def scroll(payload: ScrollRequest, manager=Depends(get_qdrant_client)) -> dict[str, Any]:
    try:
        results, next_offset = await manager.ascroll(
            filter_conditions=_to_qdrant_conditions(payload.filter_conditions),
            limit=payload.limit,
            order_by=payload.order_by,
            with_payload=payload.with_payload,
            with_vectors=payload.with_vectors,
        )
        return {
            "items": _to_plain(results),
            "count": len(results),
            "next_offset": _to_plain(next_offset),
        }
    except Exception as exc:
        logger.exception("scroll failed")
        raise QdrantApiError(f"scroll failed: {exc}") from exc


@documents_router.post("/delete/by-filter")
async def delete_by_filter(
        payload: DeleteByFilterRequest,
        manager=Depends(get_qdrant_client),
) -> dict[str, Any]:
    try:
        await manager.adelete_by_filter(_to_qdrant_conditions(payload.filter_conditions) or [])
        return {"deleted_by_filter": True}
    except Exception as exc:
        logger.exception("delete_by_filter failed")
        raise QdrantApiError(f"delete_by_filter failed: {exc}") from exc


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


@search_router.post("/groups")
async def query_points_groups(
        payload: QueryPointsGroupsRequest,
        manager=Depends(get_qdrant_client),
) -> dict[str, Any]:
    try:
        query_vector = await manager.dense_embedder.aembed_query(payload.query_text)
        groups = await manager.aquery_points_groups(
            query=query_vector,
            group_by=payload.group_by,
            limit=payload.limit,
            group_size=payload.group_size,
            filter_conditions=_to_qdrant_conditions(payload.filter_conditions),
            using=payload.using,
            with_payload=payload.with_payload,
            with_vectors=payload.with_vectors,
        )
        return _to_plain(groups)
    except Exception as exc:
        logger.exception("query_points_groups failed")
        raise QdrantApiError(f"query_points_groups failed: {exc}") from exc


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

    seen_sources: dict[str, ChatSourceModel] = {}
    ip = get_app_config().APP_IP
    port = get_app_config().APP_PORT
    url = get_app_config().APP_DOMAIN

    answer_satisfied = result.get("answer_satisfied", True)
    if answer_satisfied:
        for r in result.get("results", []):
            meta = r.get("metadata", {})
            source_file = meta.get("file_name", "")
            if source_file and source_file not in seen_sources:
                seen_sources[source_file] = ChatSourceModel(
                    name=source_file,
                    url=f"{url}/api/v1/file/{source_file}.pdf",
                    page_number=meta.get("page_number", 1),
                    total_pages=meta.get("total_pages", 1),
                )

    content = result.get("response", "") or ""
    intent = result.get("intent")

    return ChatResponse(
        success=True,
        data=ChatDataModel(
            role="assistant",
            intent=intent,
            content=content,
            sources=list(seen_sources.values()),
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


@documents_router.get("/collections")
async def get_all_collections(manager=Depends(get_qdrant_client)) -> dict:
    try:
        result = await manager.aget_all_collections()
        return _to_plain(result)
    except Exception as exc:
        logger.exception("get_all_collections failed")
        raise QdrantApiError(f"get_all_collections failed: {exc}") from exc


@documents_router.get("/collections/info")
async def get_collection_info(manager=Depends(get_qdrant_client)) -> dict:
    try:
        result = await manager.aget_collection_info()
        return _to_plain(result)
    except Exception as exc:
        logger.exception("get_collection_info failed")
        raise QdrantApiError(f"get_collection_info failed: {exc}") from exc


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


@search_router.post("/facet")
async def facet(payload: FacetRequest, manager=Depends(get_qdrant_client)) -> dict:
    try:
        result = await manager.afacet(
            key=payload.key,
            filter_conditions=_to_qdrant_conditions(payload.filter_conditions),
            exact=payload.exact,
            limit=payload.limit,
        )
        return _to_plain(result)
    except Exception as exc:
        logger.exception("facet failed")
        raise QdrantApiError(f"facet failed: {exc}") from exc


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


@file_router.get("/ingest.zip")
def download_ingest_zip():
    """Download data/ingest.zip — pre-built archive of all ingest JSON files."""
    from src.constants.app_constant import INGEST_ZIP
    if not os.path.isfile(INGEST_ZIP):
        raise NotFoundError(resource="ingest.zip", id="ingest.zip")
    return FileResponse(
        str(INGEST_ZIP),
        filename="ingest.zip",
        media_type="application/zip",
    )


@file_router.get("/source/{filename}")
def get_source_file(filename: str):
    """
    Trả về nội dung file source code trong ``src/``.

    Chỉ cần truyền tên file (vd: ``quiz_generator.py``, ``app_controller.py``).
    API sẽ tự tìm trong toàn bộ thư mục ``src/``.
    """
    from src.constants.app_constant import SRC_DIR

    if ".." in filename or "/" in filename or "\\" in filename:
        raise BadRequestError("Invalid filename — chỉ truyền tên file, không có path")

    # Tìm file trong toàn bộ src/
    matches = list(SRC_DIR.rglob(filename))
    if not matches:
        raise NotFoundError(resource="source file", id=filename)

    file_path = matches[0]

    # Đảm bảo file nằm trong src/ (chống path traversal)
    try:
        file_path.resolve().relative_to(SRC_DIR.resolve())
    except ValueError:
        raise BadRequestError("File nằm ngoài thư mục src/")

    from fastapi.responses import PlainTextResponse
    content = file_path.read_text(encoding="utf-8")
    return PlainTextResponse(content, media_type="text/plain; charset=utf-8")


@file_router.get("/{filename:path}")
def get_pdf_file(filename: str):
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise BadRequestError("Invalid file_name path")
    pdf_file = filename if filename.lower().endswith(".pdf") else filename + ".pdf"
    file_path = os.path.join(PDFS_DIR, pdf_file)
    if not os.path.isfile(file_path):
        raise NotFoundError(resource="file name not found", id=filename)
    return FileResponse(file_path, filename=pdf_file, media_type="application/pdf")


@quiz_router.post("/generate")
async def generate_quiz(payload: QuizRequest) -> dict:
    from src.core.quiz.quiz_generator import generate_quiz as _generate_quiz
    try:
        return await _generate_quiz(
            knowledge_pack=payload.knowledge_pack,
            level=payload.level.value if payload.level else None,
            level_value=payload.level_value,
            quiz_type=payload.type.value if payload.type else "random",
            rate_value=payload.rate_value,
            n=payload.total
        )
    except ValueError as exc:
        logger.exception("generate_quiz validation error")
        raise BadRequestError(str(exc)) from exc
    except Exception as exc:
        logger.exception("generate_quiz failed")
        raise QdrantApiError(f"generate_quiz failed: {exc}") from exc
