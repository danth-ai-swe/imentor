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
    "ru": "Russian",
}

UNSUPPORTED_LANGUAGE_MSG = "Ngôn ngữ này hiện không được hỗ trợ. / This language is not currently supported."


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


def get_detected_language(text: str) -> str | None:
    """
    Returns the human-readable language name if supported, None if not in supported list.
    Caller should handle None as an unsupported language signal.
    """
    code = detect_language(text)
    return _CODE_TO_NAME.get(code)  # Returns None if unsupported


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

# ---------------------------------------------------------------------------
# Main – manual test
# ---------------------------------------------------------------------------

def main():
    test_cases = [
        # (text, description)
        ("policiholder","English"),
        ("Xin chào, bạn có khỏe không?", "Vietnamese"),
        ("Hello, how are you?", "English"),
        ("こんにちは、お元気ですか？", "Japanese"),
        ("안녕하세요, 잘 지내고 계신가요?", "Korean"),
        ("Bonjour, comment allez-vous?", "French"),
        ("Hallo, wie geht es dir?", "German"),
        ("¿Hola, cómo estás?", "Spanish"),
        ("Olá, como vai você?", "Portuguese"),
        ("สวัสดี คุณเป็นยังไงบ้าง?", "Thai"),
        ("Hei, hvordan har du det?", "Norwegian (unsupported)"),
        ("Merhaba, nasılsın?", "Turkish (unsupported)"),
        ("", "Empty string edge case"),
        ("Hi", "Very short text edge case"),
    ]

    print("=" * 60)
    print(f"{'TEXT':<40} {'DESCRIPTION':<28} {'RESULT'}")
    print("=" * 60)

    for text, description in test_cases:
        result = get_detected_language(text)

        if result is None:
            output = UNSUPPORTED_LANGUAGE_MSG
        else:
            output = result

        display_text = (text[:37] + "...") if len(text) > 40 else text
        print(f"{display_text:<40} {description:<28} {output}")

    print("=" * 60)
    print("\n--- translate_to_english ---")
    samples = [
        "Xin chào thế giới",
        "こんにちは世界",
        "Hello world",
    ]
    for s in samples:
        translated = translate_to_english(s)
        print(f"  '{s}'  →  '{translated}'")


if __name__ == "__main__":
    main()
