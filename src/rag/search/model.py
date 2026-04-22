from typing import TypedDict, Dict, Any, Optional, List


class ChunkDict(TypedDict):
    id: str
    metadata: Dict[str, Any]
    text: str
    score: float


class PipelineResult(TypedDict):
    intent: Optional[str]
    response: Optional[str]
    detected_language: Optional[str]
    answer_satisfied: bool
    web_search_used: bool
    sources: List[Any]
