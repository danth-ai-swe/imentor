from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


# -------------------------
# KnowledgeBase
# -------------------------
@dataclass
class KnowledgeBase:
    id: int
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    documents: List["Document"] = field(default_factory=list)
