import requests


def translate_to_english(text: str) -> str:
    if not text or not text.strip():
        return text
    try:
        resp = requests.get(
            "https://translate.googleapis.com/translate_a/single",
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
