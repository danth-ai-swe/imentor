import base64
from functools import lru_cache

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from tenacity import wait_exponential, stop_after_attempt, retry

from src.config.app_config import _SingletonMeta, get_app_config

config = get_app_config()


def _retry_policy():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )


class _VisionClient(metaclass=_SingletonMeta):
    """Singleton wrapper around AzureChatOpenAI with vision support."""

    @staticmethod
    @lru_cache(maxsize=1)
    def get_model() -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_deployment=config.OPENAI_CHAT_MODEL,
            api_version=config.OPENAI_API_VERSION,
            azure_endpoint=config.OPENAI_API_BASE,
            temperature=0.0,
            max_tokens=config.GPT_MAX_TOKENS,
            timeout=config.GPT_TIMEOUT,
            api_key=config.OPENAI_API_KEY,
            max_retries=config.GPT_MAX_RETRIES,
        )

    @_retry_policy()
    def image_to_text(self, image_bytes: bytes, mime: str = "image/png") -> str:
        """Convert raw image bytes → descriptive text via vision model."""
        b64 = base64.b64encode(image_bytes).decode()
        llm = self.get_model()
        messages = [
            SystemMessage(
                content=(
                    "Bạn là công cụ OCR / mô tả ảnh chuyên nghiệp. "
                    "Hãy trích xuất TOÀN BỘ văn bản có trong ảnh (nếu có) "
                    "và mô tả ngắn gọn nội dung ảnh. "
                    "Trả về dạng văn bản thuần, không dùng Markdown."
                )
            ),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": "Trích xuất văn bản và mô tả ảnh trên."},
                ]
            ),
        ]
        response = llm.invoke(messages)
        return response.content.strip()

    @_retry_policy()
    async def aimage_to_text(self, image_bytes: bytes, mime: str = "image/png") -> str:
        """Async variant."""
        b64 = base64.b64encode(image_bytes).decode()
        llm = self.get_model()
        messages = [
            SystemMessage(
                content=(
                    "Bạn là công cụ OCR / mô tả ảnh chuyên nghiệp. "
                    "Hãy trích xuất TOÀN BỘ văn bản có trong ảnh (nếu có) "
                    "và mô tả ngắn gọn nội dung ảnh. "
                    "Trả về dạng văn bản thuần, không dùng Markdown."
                )
            ),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": "Trích xuất văn bản và mô tả ảnh trên."},
                ]
            ),
        ]
        response = await llm.ainvoke(messages)
        return response.content.strip()


@lru_cache(maxsize=1)
def get_vision_client() -> _VisionClient:
    return _VisionClient()
