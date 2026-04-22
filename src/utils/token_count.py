import re
from typing import Literal

import tiktoken


class TokenCount:
    def __init__(
            self,
            method: Literal["heuristic", "word", "tiktoken"] = "heuristic",
            tiktoken_encoding: str = "cl100k_base",
    ) -> None:
        self.method = method
        self.char_to_token_ratio = 0.25
        self.tiktoken_encoding = tiktoken_encoding

    def num_tokens_from_string(self, text: str) -> int:
        if not text:
            return 0

        if self.method == "word":
            return self._count_tokens_word_based(text)
        elif self.method == "tiktoken":
            return self._count_tokens_tiktoken(text)
        else:
            return self._count_tokens_heuristic(text)

    def _count_tokens_heuristic(self, text: str) -> int:
        return max(1, int(len(text) * self.char_to_token_ratio))

    def _count_tokens_tiktoken(self, text: str) -> int:
        if tiktoken is None:
            raise ImportError("tiktoken library is not installed. Please install it with 'pip install tiktoken'.")
        enc = tiktoken.get_encoding(self.tiktoken_encoding)
        return len(enc.encode(text))

    @staticmethod
    def _count_tokens_word_based(text: str) -> int:
        words = re.findall(r"\b\w+\b|\S", text)
        word_count = len(words)
        punctuation = len(re.findall(r'[.,;:!?"\']', text))
        return max(1, word_count + punctuation // 2)

    def num_tokens_from_messages(
            self,
            messages: list[dict[str, str]],
            include_overhead: bool = True,
    ) -> int:
        total = 0
        overhead_per_message = 4 if include_overhead else 0

        for msg in messages:
            content = msg.get("content", "")
            total += self.num_tokens_from_string(content) + overhead_per_message

        if include_overhead:
            total += 3

        return max(1, total)
