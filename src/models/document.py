from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


@dataclass
class Document:
    id: int
    workspace_id: int
    filename: str
    original_filename: str
    file_type: str
    file_size: int

    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    error_message: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # NexusRAG fields
    markdown_content: Optional[str] = None
    page_count: int = 0
    image_count: int = 0
    table_count: int = 0
    parser_version: Optional[str] = None
    processing_time_ms: int = 0

    images: List["DocumentImage"] = field(default_factory=list)
    tables: List["DocumentTable"] = field(default_factory=list)


# -------------------------
# DocumentImage
# -------------------------
@dataclass
class DocumentImage:
    id: int
    document_id: int
    image_id: str
    page_no: int = 0
    file_path: str = ""
    caption: str = ""
    width: int = 0
    height: int = 0
    mime_type: str = "image/png"
    created_at: datetime = field(default_factory=datetime.utcnow)


# -------------------------
# DocumentTable
# -------------------------
@dataclass
class DocumentTable:
    id: int
    document_id: int
    table_id: str
    page_no: int = 0
    content_markdown: str = ""
    caption: str = ""
    num_rows: int = 0
    num_cols: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
