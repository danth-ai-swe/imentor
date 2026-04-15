import json
import re

import unicodedata

from src.constants.app_constant import QUIZ_KEYWORDS


def cosine_similarity(vec1: list, vec2: list) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def strip_json_fence(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def parse_llm_json(raw: str) -> list[dict]:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(cleaned)
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    return data if isinstance(data, list) else [data]


def normalize_ellipsis(text: str, max_dots: int = 3) -> str:
    """Chuẩn hóa dấu chấm lửng thừa"""
    pattern = r'\.{' + str(max_dots + 1) + r',}'
    return re.sub(pattern, '.' * max_dots, text)


def is_quiz_intent(text: str) -> bool:
    normalised = text.lower().strip()
    return normalised in QUIZ_KEYWORDS or any(kw in normalised for kw in QUIZ_KEYWORDS)


def parse_json_response(raw: str) -> dict:
    match = re.search(r"\{.*?}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group(0))


def clean_text(text: str) -> str:
    """
    Pipeline làm sạch văn bản hoàn chỉnh:
    1. Chuẩn hóa unicode
    2. Xóa ký tự đặc biệt thừa
    3. Chuẩn hóa dấu chấm lửng
    4. Chuẩn hóa khoảng trắng
    5. Xóa dòng trống thừa
    """
    # 1. Chuẩn hóa unicode (NFC)
    text = unicodedata.normalize("NFC", text)

    # 2. Xóa ký tự không in được (trừ newline, tab)
    text = re.sub(r'[^\S\n\t ]+', ' ', text)  # Ký tự khoảng trắng lạ
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # Control chars

    # 3. Chuẩn hóa dấu chấm lửng (...... → ...)
    text = normalize_ellipsis(text, max_dots=3)

    # 4. Chuẩn hóa dấu câu lặp (!!!! → !, ???? → ?)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'-{3,}', '—', text)  # --- → em dash

    # 5. Xóa khoảng trắng thừa trong dòng
    text = re.sub(r'[ \t]+', ' ', text)

    # 6. Xóa dòng trống liên tiếp (> 2 dòng trống → 1 dòng trống)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 7. Trim từng dòng và toàn bộ văn bản
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(lines).strip()

    return text
