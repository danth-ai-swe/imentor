# language_utils.py
# Toàn bộ logic detect/translate qua Google Translate API bị xoá.
# Language detection giờ được xử lý bởi LLM trong pipeline.
# File này chỉ giữ lại constants dùng chung.
import requests

SUPPORTED_LANGUAGES = {"English", "Vietnamese", "Japanese"}
lang_map = {
    "English": "en",
    "Vietnamese": "vi",
    "Japanese": "ja"
}

_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"


def _translate(text: str, target_lang: str) -> str:
    try:
        resp = requests.get(
            _TRANSLATE_URL,
            params={
                "client": "gtx",
                "sl": "auto",
                "tl": target_lang,
                "dt": "t",
                "q": text,
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()

        translated_parts = [
            part[0] for part in data[0] if part and part[0]
        ]
        return "".join(translated_parts)
    except Exception:
        return text


def is_english(text: str) -> bool:
    return detect_language(text) == "en"


def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 3:
        return "en"
    try:
        resp = requests.get(
            _TRANSLATE_URL,
            params={
                "client": "gtx",
                "sl": "auto",
                "tl": "en",
                "dt": "t",
                "q": text[:300],
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        detected = data[2] if len(data) > 2 else None
        if detected and isinstance(detected, str) and detected != "auto":
            return detected
        return "en"
    except Exception:
        return "en"
