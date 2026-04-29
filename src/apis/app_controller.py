import json
import os
from datetime import datetime, timezone

from fastapi import BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

from src.apis.app_exception import BadRequestError, NotFoundError, QdrantApiError
from src.apis.app_model import (
    ChatRequest,
    ChatResponse,
    ChatDataModel,
    QuizRequest
)
from src.apis.app_router import documents_router, chat_router, file_router, quiz_router
from src.constants.app_constant import (
    PDFS_DIR, COLLECTION_NAME, T_ZIP, TT_ZIP, STORAGE_FAQ_TEMPLATE,
    OVERALL_COLLECTION_NAME, OVERALL_INGEST_DIR, INGEST_DIR,
)
from src.core.quiz.quiz_generator import generate_quiz, generate_quiz_full_background, read_quiz_result
from src.rag.db_vector import get_qdrant_client
from src.rag.ingest.pipeline import upload_to_qdrant
from src.rag.search.pipeline import async_pipeline_dispatch, async_pipeline_dispatch_stream
from src.rag.search.agent.streaming import agent_stream_events
from src.utils.logger_utils import logger


@documents_router.get("/ingest")
async def ingest_documents(force_restart: bool = False, collection_name: str | None = None,
                           background_tasks: BackgroundTasks = None):
    target_collection = collection_name or COLLECTION_NAME
    source_dir = (
        OVERALL_INGEST_DIR
        if target_collection == OVERALL_COLLECTION_NAME
        else INGEST_DIR
    )
    try:
        if not force_restart:
            manager = get_qdrant_client(target_collection)
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

    background_tasks.add_task(
        upload_to_qdrant,
        force_restart=force_restart,
        collection_name=target_collection,
        source_dir=source_dir,
    )
    return {"status": "started",
            "message": f"Ingest triggered in background for collection '{target_collection}' from '{source_dir}'."}


@chat_router.post("/ask", response_model=ChatResponse)
async def chat_ask(payload: ChatRequest) -> ChatResponse:
    logger.info("chat_ask called | user_name=%s | conversation_id=%s", payload.user_name, payload.conversation_id)
    try:
        result = await async_pipeline_dispatch(
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
            web_search_used=result.get("web_search_used", False),
        ),
    )


@chat_router.post("/ask/stream")
async def chat_ask_stream(payload: ChatRequest) -> StreamingResponse:
    """SSE endpoint: streams answer tokens as they are generated.

    Event format (SSE data payload is JSON):
      event: meta  → {intent, detected_language, sources, answer_satisfied, web_search_used}
      event: delta → {content: "<token piece>"}
      event: done  → {}
    """
    logger.info(
        "chat_ask_stream called | user_name=%s | conversation_id=%s",
        payload.user_name, payload.conversation_id,
    )

    async def event_source():
        try:
            async for ev in async_pipeline_dispatch_stream(
                user_input=payload.message,
                conversation_id=payload.conversation_id,
            ):
                event_type = ev.get("type", "message")
                data = {k: v for k, v in ev.items() if k != "type"}
                yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("chat_ask_stream failed")
            err = json.dumps({"message": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@chat_router.post("/ask/stream/agent")
async def chat_ask_stream_agent(payload: ChatRequest) -> StreamingResponse:
    """SSE endpoint for the LangGraph hybrid agent.

    Same event shape as /ask/stream:
      event: meta  → {intent, detected_language, sources, answer_satisfied,
                      web_search_used, tool_call_count}
      event: delta → {content: "<token piece>"}
      event: done  → {}
      event: error → {message: "<exc str>"}  (on uncaught failure)
    """
    logger.info(
        "chat_ask_stream_agent called | user_name=%s | conversation_id=%s",
        payload.user_name, payload.conversation_id,
    )

    async def event_source():
        try:
            async for ev in agent_stream_events(
                user_input=payload.message,
                conversation_id=payload.conversation_id,
            ):
                event_type = ev.get("type", "message")
                data = {k: v for k, v in ev.items() if k != "type"}
                yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("chat_ask_stream_agent failed")
            err = json.dumps({"message": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@file_router.get("/t")
def get_t_zip_file():
    if not os.path.isfile(T_ZIP):
        raise NotFoundError(resource="T file not found", id=T_ZIP)
    return FileResponse(T_ZIP, filename=os.path.basename(T_ZIP), media_type="application/zip")


@file_router.get("/tt")
def get_tt_zip_file():
    if not os.path.isfile(TT_ZIP):
        raise NotFoundError(resource="TT file not found", id=TT_ZIP)
    return FileResponse(TT_ZIP, filename=os.path.basename(TT_ZIP), media_type="application/zip")


@file_router.get("/faq-template")
def get_fqa_template():
    if not os.path.isfile(STORAGE_FAQ_TEMPLATE):
        raise NotFoundError(resource="STORAGE_FAQ_TEMPLATE file not found", id=STORAGE_FAQ_TEMPLATE)
    response = FileResponse(STORAGE_FAQ_TEMPLATE, filename=os.path.basename(STORAGE_FAQ_TEMPLATE),
                            media_type="application/pdf")
    response.headers["Content-Disposition"] = 'inline; filename="iMentor_FAQ_Template.pdf"'
    return response


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
async def generate(payload: QuizRequest) -> dict:
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


@quiz_router.post("/generate-full")
async def generate_full(knowledge_pack: str, background_tasks: BackgroundTasks) -> dict:
    try:
        background_tasks.add_task(
            generate_quiz_full_background,
            knowledge_pack=knowledge_pack
        )
        return {
            "success": True,
            "message": f"Đang sinh câu hỏi cho '{knowledge_pack}' ở background. "
        }
    except Exception as exc:
        logger.exception("generate_quiz_full failed")
        raise QdrantApiError(f"generate_quiz_full failed: {exc}") from exc


@quiz_router.get("/generate-full/result")
async def get_generate_full_result(
        knowledge_pack: str
) -> dict:
    try:
        return read_quiz_result(knowledge_pack)
    except Exception as exc:
        logger.exception("read_quiz_result failed")
        raise QdrantApiError(f"read_quiz_result failed: {exc}") from exc


from src.core.quiz.quiz_chat.model import QuizChatRequest, QuizChatResponse
from src.core.quiz.quiz_chat.pipeline import async_quiz_chat_dispatch


@quiz_router.post("/chat", response_model=QuizChatResponse)
async def quiz_chat(payload: QuizChatRequest) -> QuizChatResponse:
    logger.info(
        "quiz_chat called | node_name=%s | message_len=%d",
        payload.current_question.node_name, len(payload.message),
    )
    try:
        return await async_quiz_chat_dispatch(payload)
    except Exception as exc:
        logger.exception("quiz_chat failed")
        raise QdrantApiError(f"quiz_chat failed: {exc}") from exc
