import json

import pandas as pd

from src.rag.clean_data.model import Chunk
from src.rag.clean_data.prompt import CLASSIFY_NODE_PROMPT
from src.rag.llm.chat_llm import get_openai_chat_client
from src.utils.app_utils import strip_json_fence
from src.utils.logger_utils import logger


class NodeClassifier:
    """Phân loại chunk vào node theo metadata_node.xlsx."""

    def __init__(self, xlsx_path: str) -> None:
        self.xlsx_path = xlsx_path
        self.client = get_openai_chat_client()
        self._cache: dict[str, pd.DataFrame] = {}

    def _get_candidates(self, source_prefix: str) -> pd.DataFrame:
        if source_prefix not in self._cache:
            df = pd.read_excel(self.xlsx_path)
            mask = df["Source"].astype(str).str.startswith(source_prefix)
            self._cache[source_prefix] = (
                df[mask][["Node ID", "Node Name", "Category"]].drop_duplicates()
            )
        return self._cache[source_prefix]

    def classify(self, chunk: Chunk) -> None:
        """Gán node_id / node_name / category trực tiếp vào chunk (in-place)."""
        parts = chunk.file_name.split("_")
        source_prefix = "_".join(parts[:2])

        candidates_df = self._get_candidates(source_prefix)
        if candidates_df.empty:
            return  # fields giữ None mặc định

        candidates_str = "\n".join(
            f"{row['Node ID']} | {row['Category']} | {row['Node Name']}"
            for _, row in candidates_df.iterrows()
        )
        prompt = CLASSIFY_NODE_PROMPT.format(
            candidates=candidates_str,
            chunk_text=chunk.text[:2000],
        )
        raw = strip_json_fence(self.client.invoke(prompt))
        try:
            result = json.loads(raw)
            chunk.node_id = result.get("node_id")
            chunk.node_name = result.get("node_name")
            chunk.category = result.get("category")
        except json.JSONDecodeError:
            logger.warning("Không parse được JSON từ classifier cho chunk %d", chunk.chunk_index)

    def classify_all(self, chunks: list[Chunk]) -> None:
        """Phân loại toàn bộ danh sách chunk (in-place)."""
        total = len(chunks)
        for chunk in chunks:
            logger.info(
                "  Classifying chunk %d/%d (page %d)…",
                chunk.chunk_index + 1, total, chunk.page_number,
            )
            self.classify(chunk)
