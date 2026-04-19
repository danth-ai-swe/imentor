from enum import Enum
from typing import Any, List

from pydantic import BaseModel, Field

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
    web_search_used: bool = False


class ChatResponse(BaseModel):
    success: bool
    data: ChatDataModel


class UploadDocumentsRequest(BaseModel):
    documents: list[dict]
    batch_size: int = 256
    parallel: int = 4
    max_retries: int = 3


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
    total: int = Field(..., ge=1, le=1000, description="Number of questions to generate")
