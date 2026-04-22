import re
from functools import lru_cache
from typing import Any, List, Optional

from src.rag.chunking.fixed_token_chunker import TextSplitter


def _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> List[str]:
    if separator:
        if keep_separator:
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class RecursiveTokenChunker(TextSplitter):

    def __init__(
            self,
            chunk_size: int = 4000,
            chunk_overlap: int = 200,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = False,
            **kwargs: Any,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=keep_separator,
            **kwargs,
        )
        self._separators = separators or ["\n\n", "\n", ".", "?", "!", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []

        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        good_splits: List[str] = []
        merge_separator = "" if self._keep_separator else separator

        for split in splits:
            if self._length_function(split) < self._chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, merge_separator))
                    good_splits = []
                if not new_separators:
                    final_chunks.append(split)
                else:
                    final_chunks.extend(self._split_text(split, new_separators))

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, merge_separator))

        return final_chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)


@lru_cache(maxsize=1)
def get_recursive_token_chunk(chunk_size: int = 800) -> RecursiveTokenChunker:
    separators = [
        "\n# ",
        "\n## ",
        "\n### ",
        "\n```\n",
        "\n---\n",
        "\n***\n",
        "\n\n\n",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ]
    return RecursiveTokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.2),
        separators=separators,
        keep_separator=True,
        is_separator_regex=False,
    )
