import re
from functools import lru_cache
from typing import List

from src.rag.chunking.base_chunker import BaseChunker
from src.rag.chunking.recursive_chunker import RecursiveTokenChunker
from src.rag.llm.chat_llm import get_openai_chat_client
from src.utils.logger_utils import logger
from src.utils.token_count import TokenCount

_MAX_WINDOW_TOKENS = 800


class LLMAgenticChunkerv2(BaseChunker):

    def __init__(self) -> None:
        self.client = get_openai_chat_client()
        self.token_counter = TokenCount(method="heuristic")
        self.splitter = RecursiveTokenChunker(
            chunk_size=50,
            chunk_overlap=0,
            length_function=self.token_counter.num_tokens_from_string,
        )

    @staticmethod
    def _build_messages(
            chunked_input: str,
            current_chunk: int,  # 1-indexed chunk ID (matches prompt IDs)
            max_chunk: int,
    ) -> List[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You are an assistant specialized in splitting text into thematically "
                    "consistent sections. The text has been divided into chunks, each marked "
                    "with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
                    "Your task is to identify the points where splits should occur, such that "
                    "consecutive chunks of similar themes stay together. "
                    f"IMPORTANT: You MUST use the EXACT chunk numbers shown in the <|start_chunk_X|> tags. "
                    f"The chunks in this window start at {current_chunk} and end at {max_chunk} — do NOT use relative positions like 1, 2, 3. "
                    f"Respond ONLY with: 'split_after: {current_chunk}, {current_chunk + 2}' "
                    f"(use actual chunk IDs from the tags, in strictly ascending order)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CHUNKED_TEXT: {chunked_input}\n\n"
                    f"Respond only with the chunk IDs where a split should occur. "
                    f"YOU MUST RESPOND WITH AT LEAST ONE SPLIT. "
                    f"All IDs must be EXACTLY as shown in the <|start_chunk_X|> tags, "
                    f"in strictly ascending order, and must be between {current_chunk} and {max_chunk}."
                ),
            },
        ]

    def _get_split_points(
            self,
            chunked_input: str,
            current_chunk: int,
            max_chunk: int
    ) -> List[int]:
        try:
            result = self.client.chat(
                self._build_messages(chunked_input, current_chunk, max_chunk)
            )
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return []

        lines_with_split = [l for l in result.split("\n") if "split_after:" in l]
        if lines_with_split:
            numbers = list(map(int, re.findall(r"\d+", lines_with_split[0])))
        else:
            numbers = list(map(int, re.findall(r"\d+", result)))
            if not numbers:
                logger.warning(f"No parseable numbers in response: {result!r}")
                return []

        if numbers and max(numbers) < current_chunk:
            offset = current_chunk - 1
            logger.warning(
                f"Detected relative IDs {numbers}, "
                f"shifting by offset {offset} → {[n + offset for n in numbers]}"
            )
            numbers = [n + offset for n in numbers]

        is_ascending = numbers == sorted(set(numbers))
        all_valid = all(current_chunk <= n <= max_chunk for n in numbers)
        has_values = len(numbers) > 0

        if is_ascending and all_valid and has_values:
            return sorted(set(numbers))

        logger.warning(
            f"Invalid split response: {numbers} "
            f"(ascending={is_ascending}, all_valid={all_valid}, "
            f"expected {current_chunk}..{max_chunk})"
        )
        return []

    def split_text(self, text: str) -> List[str]:
        if not text or len(text.strip()) < 50:
            return [text.strip()] if text.strip() else []

        chunks: List[str] = self.splitter.split_text(text)
        if len(chunks) <= 1:
            return chunks

        split_indices: List[int] = []
        current_chunk = 0
        while current_chunk < len(chunks) - 4:
            token_count = 0
            chunked_input = ""
            window_end_idx = current_chunk
            for i in range(current_chunk, len(chunks)):
                token_count += self.token_counter.num_tokens_from_string(chunks[i])
                chunked_input += f"<|start_chunk_{i + 1}|>{chunks[i]}<|end_chunk_{i + 1}|>"
                window_end_idx = i
                if token_count > _MAX_WINDOW_TOKENS:
                    break

            window_start_id = current_chunk + 1
            window_end_id = window_end_idx + 1

            numbers = self._get_split_points(chunked_input, window_start_id, window_end_id)

            if not numbers:
                mid = (current_chunk + window_end_idx) // 2
                fallback_split_id = mid + 1  # convert to 1-indexed
                logger.warning(
                    f"LLM failed entirely — using fallback midpoint split at chunk {fallback_split_id} "
                    f"(window {window_start_id}..{window_end_id})"
                )
                split_indices.append(fallback_split_id)
                current_chunk = fallback_split_id  # 1-indexed, matches loop bound
                continue

            if numbers[-1] <= current_chunk:
                logger.warning(
                    f"Split point {numbers[-1]} does not advance past {current_chunk}, forcing skip"
                )
                current_chunk = min(current_chunk + 4, len(chunks) - 1)
                continue

            split_indices.extend(numbers)
            current_chunk = numbers[-1]

        chunks_to_split_after = {i - 1 for i in split_indices}

        docs: List[str] = []
        current_doc = ""

        for i, chunk in enumerate(chunks):
            current_doc += chunk + " "
            if i in chunks_to_split_after:
                stripped = current_doc.strip()
                if stripped:
                    docs.append(stripped)
                current_doc = ""

        if current_doc.strip():
            docs.append(current_doc.strip())

        return docs if docs else [text]


@lru_cache(maxsize=1)
def get_agentic_chunker() -> LLMAgenticChunkerv2:
    return LLMAgenticChunkerv2()
