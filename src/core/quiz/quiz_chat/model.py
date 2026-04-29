"""Pydantic models for the /quiz/chat endpoint."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.apis.app_model import ChatSourceModel


class OptionItem(BaseModel):
    index: Literal["A", "B", "C", "D"]
    text: str = Field(..., min_length=1)


class CurrentQuestion(BaseModel):
    question_type: Optional[str] = None
    difficulty: Optional[str] = None
    question: str = Field(..., min_length=1)
    options: List[OptionItem] = Field(..., min_length=4, max_length=4)
    category: str = Field(..., min_length=1)
    node_name: str = Field(..., min_length=1)
    sources: Optional[List[dict]] = None


class QuizChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=15_000)
    current_question: CurrentQuestion


class QuizChatResponse(BaseModel):
    intent: Literal["hint", "finish", "answer", "question"]
    answer_index: Optional[Literal["A", "B", "C", "D"]] = None
    message: Optional[str] = None
    content: Optional[str] = None
    sources: Optional[List[ChatSourceModel]] = None
