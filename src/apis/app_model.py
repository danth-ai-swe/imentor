from typing import Any, List

from pydantic import BaseModel, Field


class FilterConditionModel(BaseModel):
    key: str = Field(..., description="Payload field name")
    value: Any = Field(..., description="Exact value to match")


class GetByIdsRequest(BaseModel):
    ids: list[int | str] = Field(..., min_length=1)
    with_vectors: bool = False
    with_payload: bool = True


class ScrollRequest(BaseModel):
    filter_conditions: list[FilterConditionModel] | None = None
    limit: int = Field(15, ge=1, le=100)
    order_by: str | None = None
    with_payload: bool = True
    with_vectors: bool = False


class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=100)
    prefetch_limit: int = Field(50, ge=1, le=500)
    filter_conditions: list[FilterConditionModel] | None = None
    with_payload: bool = True


class BuildFilterRequest(BaseModel):
    must: list[FilterConditionModel] | None = None
    should: list[FilterConditionModel] | None = None
    must_not: list[FilterConditionModel] | None = None
    min_should: Any | None = None


class QueryPointsGroupsRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    group_by: str = Field(..., min_length=1)
    limit: int = Field(2, ge=1, le=100)
    group_size: int = Field(2, ge=1, le=100)
    filter_conditions: list[FilterConditionModel] | None = None
    using: str | None = "dense"
    with_payload: bool = True
    with_vectors: bool = False


class DeleteByIdsRequest(BaseModel):
    ids: list[int | str] = Field(..., min_length=1)


class DeleteByFilterRequest(BaseModel):
    filter_conditions: list[FilterConditionModel] = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=15_000, description="User message")
    user_name: str = Field(..., min_length=1, description="Display name of the user")
    conversation_id: str | None = Field(default=None, description="Conversation / chat session ID (optional)")


class ChatSourceModel(BaseModel):
    name: str
    url: str
    page_number: int
    total_pages: int


class ChatDataModel(BaseModel):
    role: str = "assistant"
    intent: str | None = None
    content: str
    sources: List[ChatSourceModel]
    timestamp: str


class ChatResponse(BaseModel):
    success: bool
    data: ChatDataModel


class IntentRouteRequest(BaseModel):
    texts: str | list[str]


class IntentRouteResponse(BaseModel):
    results: list[dict]


class GenerateEmbeddingsResponse(BaseModel):
    routes: list[dict]
    output_path: str


class UploadDocumentsRequest(BaseModel):
    documents: list[dict]
    batch_size: int = 256
    parallel: int = 4
    max_retries: int = 3


class FacetRequest(BaseModel):
    key: str
    filter_conditions: list[FilterConditionModel] | None = None
    exact: bool = True
    limit: int = 10


class BatchSearchRequest(BaseModel):
    queries: list[str]
    top_k: int = 5
    prefetch_limit: int = 50


class CreatePayloadIndexRequest(BaseModel):
    field_name: str
    field_schema: str


# ── Quiz models ──────────────────────────────────────────────────────────────

from enum import Enum
from pydantic import model_validator


class QuizLevel(str, Enum):
    MODULE = "module"
    LESSON = "lesson"
    COURSE = "course"


class QuizDifficulty(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


class QuizRequest(BaseModel):
    knowledge_pack: str = Field(..., min_length=1, description="Always compared with metadata 'course'")
    module_value: str | None = Field(None, description="Value to match against metadata 'Module' column")
    lesson_value: str | None = Field(None, description="Value to match against metadata 'Lesson' column")
    difficulty: QuizDifficulty | None = Field(
        None,
        description="Difficulty level: Beginner, Intermediate, or Advanced. If null, will be randomized."
    )
    total: int = Field(..., ge=1, le=200, description="Number of questions to generate")
