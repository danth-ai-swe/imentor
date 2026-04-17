import base64
import json
from typing import Optional

import fitz

from rag.clean_data.prompt import EXTRACT_TABLE_PROMPT, EXTRACT_CHART_PROMPT, EXTRACT_TEXT_PROMPT, CLASSIFY_PROMPT
from utils.app_utils import strip_json_fence
from utils.logger_utils import logger

MIN_IMAGE_PIXELS = 5_000
_LABEL_MAP = {
    "table": "**[TABLE]**",
    "chart": "**[CHART]**",
    "text_data": "**[TEXT DATA]**",
}
_EXTRACT_PROMPT_MAP = {
    "table": EXTRACT_TABLE_PROMPT,
    "chart": EXTRACT_CHART_PROMPT,
    "text_data": EXTRACT_TEXT_PROMPT,
}


class ImageProcessor:
    """Phân loại và trích xuất nội dung từ ảnh nhúng trong PDF."""

    def __init__(self, client) -> None:
        self.client = client

    @staticmethod
    def pixmap_to_base64(pix: fitz.Pixmap) -> str:
        return base64.b64encode(pix.tobytes("png")).decode("utf-8")

    def classify(self, b64: str) -> dict:
        raw = self.client.invoke_with_image(
            prompt=CLASSIFY_PROMPT,
            image_base64=b64,
            media_type="image/png",
        )
        raw = strip_json_fence(raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"type": "other", "reason": raw}

    def extract(self, b64: str, img_type: str) -> str:
        prompt = _EXTRACT_PROMPT_MAP.get(img_type)
        if not prompt:
            return ""
        return self.client.invoke_with_image(
            prompt=prompt,
            image_base64=b64,
            media_type="image/png",
        )

    def process_image(self, doc: fitz.Document, xref: int) -> Optional[str]:
        """
        Xử lý một ảnh: phân loại rồi trích xuất nội dung.
        Trả về chuỗi markdown hoặc None nếu bỏ qua.
        """
        try:
            base_image = doc.extract_image(xref)
        except Exception as exc:
            logger.warning("Không trích xuất được ảnh xref=%s: %s", xref, exc)
            return None

        w, h = base_image.get("width", 0), base_image.get("height", 0)
        if w * h < MIN_IMAGE_PIXELS:
            logger.debug("Bỏ qua ảnh nhỏ (%dx%dpx)", w, h)
            return None

        logger.debug("Xử lý ảnh xref=%s (%dx%dpx)…", xref, w, h)
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha > 3:  # normalize CMYK → RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)

        b64 = self.pixmap_to_base64(pix)
        classification = self.classify(b64)
        img_type = classification.get("type", "other")
        logger.debug("→ Loại: %s | %s", img_type, classification.get("reason", ""))

        if img_type == "other":
            return None

        content = self.extract(b64, img_type)
        label = _LABEL_MAP[img_type]
        return f"{label}\n\n{content}"
