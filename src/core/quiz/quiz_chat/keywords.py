"""Keyword sets and text normalisation for the quiz-chat intent classifier."""
import re
import unicodedata


def normalise(text: str) -> str:
    """Lowercase, NFC-normalise, collapse whitespace, strip leading/trailing punctuation."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.strip(".,!?;:'\"()[]{}")
    text = text.strip()
    return text


FINISH_KEYWORDS: frozenset[str] = frozenset({
    # English
    "finish", "finish session", "end", "end session", "end quiz",
    "submit", "submit and finish", "done", "quit", "exit", "stop",
    # Vietnamese (with and without diacritics)
    "kết thúc", "ket thuc",
    "thoát", "thoat",
    "dừng", "dung",
    "nộp bài", "nop bai",
})


HINT_KEYWORDS: frozenset[str] = frozenset({
    # English
    "hint", "hin", "give me a hint", "help",
    # Vietnamese
    "gợi ý", "goi y",
})
