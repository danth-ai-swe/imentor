import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Any, Dict


# -------------------------
# ChatMessage
# -------------------------
@dataclass
class ChatMessage:
    id: int
    workspace_id: int
    message_id: str
    role: str
    content: str

    sources: Optional[List[Any]] = None
    related_entities: Optional[List[Any]] = None
    image_refs: Optional[List[Any]] = None
    thinking: Optional[str] = None
    ratings: Optional[Dict[str, Any]] = None
    agent_steps: Optional[List[Any]] = None

    created_at: datetime = field(default_factory=datetime.utcnow)


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PARSING = "parsing"
    PROCESSING = "processing"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
