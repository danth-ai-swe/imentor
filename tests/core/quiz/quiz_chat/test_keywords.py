from src.core.quiz.quiz_chat.keywords import (
    FINISH_KEYWORDS,
    HINT_KEYWORDS,
    normalise,
)


def test_normalise_strips_and_lowercases():
    assert normalise("  Hello!  ") == "hello"
    assert normalise("KẾT THÚC.") == "kết thúc"
    assert normalise("\thint?\n") == "hint"


def test_normalise_collapses_internal_whitespace():
    assert normalise("end   quiz") == "end quiz"
    assert normalise("submit\tand\nfinish") == "submit and finish"


def test_finish_keywords_contain_required_phrases():
    required = {
        "finish", "finish session", "end", "end session", "end quiz",
        "submit and finish", "done", "quit", "exit", "stop",
        "kết thúc", "thoát", "dừng", "nộp bài",
    }
    assert required.issubset(FINISH_KEYWORDS)


def test_hint_keywords_contain_required_phrases():
    required = {
        "hint", "hin", "give me a hint", "help",
        "gợi ý", "goi y",
    }
    assert required.issubset(HINT_KEYWORDS)


def test_keyword_sets_are_lowercase_and_normalised():
    for kw in FINISH_KEYWORDS | HINT_KEYWORDS:
        assert kw == normalise(kw), f"{kw!r} is not pre-normalised"
