import httpx

_GOOGLE_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"


def translate_to_english(text: str) -> str:
    """Synchronous fallback kept for non-async callers (e.g. ingestion)."""
    if not text or not text.strip():
        return text
    try:
        resp = httpx.get(
            _GOOGLE_TRANSLATE_URL,
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
        return _parse_translate_response(resp.json(), text)
    except Exception:
        return text


async def atranslate_to_english(text: str) -> str:
    """Async translation using httpx — does not block the event loop."""
    if not text or not text.strip():
        return text
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                _GOOGLE_TRANSLATE_URL,
                params={
                    "client": "gtx",
                    "sl": "auto",
                    "tl": "en",
                    "dt": "t",
                    "q": text,
                },
            )
            resp.raise_for_status()
            return _parse_translate_response(resp.json(), text)
    except Exception:
        return text


def _parse_translate_response(data, original: str) -> str:
    detected = data[2] if len(data) > 2 else "en"
    if detected == "en":
        return original
    parts = [part[0] for part in data[0] if part and part[0]]
    translated = "".join(parts)
    return translated or original
