from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Chunk:
    text: str
    file_name: str
    page_number: int
    total_pages: int
    total_chunks: int
    chunk_index: int
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    category: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
