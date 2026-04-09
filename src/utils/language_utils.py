import requests

_DETECT_URL = "https://translate.googleapis.com/translate_a/single"

_CODE_TO_NAME = {
    "en": "English",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-CN": "Chinese",
    "zh-TW": "Chinese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "ar": "Arabic",
    "hi": "Hindi",
    "ru": "Russian"
}


def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 3:
        return "en"
    try:
        resp = requests.get(
            _DETECT_URL,
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


def is_english(text: str) -> bool:
    return detect_language(text) == "en"


def translate_to_english(text: str) -> str:
    if not text or not text.strip():
        return text
    try:
        resp = requests.get(
            _DETECT_URL,
            params={
                "client": "gtx",
                "sl": "auto",
                "tl": "en",
                "dt": "t",
                "q": text,
            },
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        detected = data[2] if len(data) > 2 else "en"

        if detected == "en":
            return text

        translated_parts = [
            part[0] for part in data[0] if part and part[0]
        ]
        translated = "".join(translated_parts)
        return translated if translated else text
    except Exception:
        return text


def language_name(code: str) -> str:
    return _CODE_TO_NAME.get(code, code.capitalize())
