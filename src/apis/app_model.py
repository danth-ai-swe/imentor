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


class QuizType(str, Enum):
    RANDOM = "random"
    RATE = "rate"


class QuizRequest(BaseModel):
    knowledge_pack: str = Field(..., min_length=1, description="Always compared with metadata 'course'")
    total: int = Field(..., ge=1, le=200, description="Number of questions to generate")
    level: QuizLevel | None = Field(None, description="Additional metadata field to filter on: module, lesson, or course")
    level_value: str | None = Field(None, description="Value to match against metadata[level]")
    type: QuizType | None = Field(None, description="'random' = random difficulty ratio; 'rate' = use rate_value to set ratio. Default: random")
    rate_value: str | None = Field(None, description="Difficulty ratio as 'B|I|A' e.g. '2|5|3' → Beginner 20%, Intermediate 50%, Advanced 30%. Required when type='rate'.")

    @model_validator(mode="after")
    def _validate_fields(self):
        # level & level_value must come together
        has_level = self.level is not None
        has_value = self.level_value is not None and self.level_value.strip() != ""
        if has_level != has_value:
            raise ValueError("'level' and 'level_value' must be provided together or both omitted")

        # default type to RANDOM when not provided
        if self.type is None:
            self.type = QuizType.RANDOM

        # rate_value is required when type='rate'
        if self.type == QuizType.RATE:
            if not self.rate_value or not self.rate_value.strip():
                raise ValueError("'rate_value' is required when type is 'rate'")
            parts = self.rate_value.strip().split("|")
            if len(parts) != 3:
                raise ValueError("'rate_value' must have exactly 3 parts separated by '|', e.g. '2|5|3'")
            for p in parts:
                if not p.strip().isdigit() or int(p.strip()) < 0:
                    raise ValueError(f"Each part of 'rate_value' must be a non-negative integer, got '{p}'")
            if sum(int(p.strip()) for p in parts) == 0:
                raise ValueError("'rate_value' parts must not all be zero")
        return self