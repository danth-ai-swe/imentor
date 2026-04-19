import threading
from typing import List

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.config.app_config import retry_policy, get_app_config
from src.utils.logger_utils import alog_method_call, log_method_call

config = get_app_config()

_chat_lock = threading.Lock()
_chat_client_instance: "AzureChatClient | None" = None

_langfuse = Langfuse(
    public_key=config.LANGFUSE_PUBLIC_KEY,
    secret_key=config.LANGFUSE_SECRET_KEY,
    host=config.LANGFUSE_BASE_URL,
)

# ✅ Fix type error: RunnableConfig thay vì dict, CallbackHandler() không tham số
_lf_handler = CallbackHandler()
_lf_config: RunnableConfig = RunnableConfig(callbacks=[_lf_handler])


# ── AzureChatClient ────────────────────────────────────────────────────────────
class AzureChatClient:

    def __init__(self):
        self._llm = self._build_llm()

    @staticmethod
    def _build_llm() -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_endpoint=config.OPENAI_API_BASE,
            api_key=config.OPENAI_API_KEY,
            api_version=config.OPENAI_API_VERSION,
            azure_deployment=config.OPENAI_CHAT_MODEL,
            temperature=config.GPT_TEMPERATURE,
            top_p=config.GPT_TOP_P,
            max_tokens=config.GPT_MAX_TOKENS,
            timeout=config.GPT_TIMEOUT,
            max_retries=config.GPT_MAX_RETRIES,
        )

    @log_method_call
    @retry_policy()
    def invoke(self, prompt: str) -> str:
        # ✅ Set input trên observation → hiện trong UI
        with _langfuse.start_as_current_observation(name="invoke", input=prompt) as obs:
            response = self._llm.invoke(
                [HumanMessage(content=prompt)],
                config=_lf_config,  # ✅ RunnableConfig, không phải dict
            )
            content = response.content or ""
            obs.update(output=content)  # ✅ Set output → hiện trong UI
        return content

    @log_method_call
    @retry_policy()
    def invoke_with_image(
            self,
            prompt: str,
            image_base64: str,
            media_type: str = "image/jpeg",
    ) -> str:
        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
            {"type": "text", "text": prompt},
        ])
        with _langfuse.start_as_current_observation(name="invoke_with_image", input=prompt) as obs:
            response = self._llm.invoke([message], config=_lf_config)
            content = response.content or ""
            obs.update(output=content)
        return content

    @alog_method_call
    @retry_policy()
    async def ainvoke(self, prompt: str) -> str:
        with _langfuse.start_as_current_observation(name="ainvoke", input=prompt) as obs:
            response = await self._llm.ainvoke(
                [HumanMessage(content=prompt)],
                config=_lf_config,
            )
            content = response.content or ""
            obs.update(output=content)
        return content

    @log_method_call
    @retry_policy()
    def chat(
            self,
            messages: List[BaseMessage],
            max_tokens: int | None = None,
    ) -> str:
        llm = self._llm.bind(max_tokens=max_tokens) if max_tokens else self._llm
        input_text = str([m.content for m in messages])
        with _langfuse.start_as_current_observation(name="chat", input=input_text) as obs:
            response = llm.invoke(messages, config=_lf_config)
            content = response.content or ""
            obs.update(output=content)
        return content

    @alog_method_call
    @retry_policy()
    async def achat(
            self,
            messages: List[BaseMessage],
            max_tokens: int | None = None,
    ) -> str:
        llm = self._llm.bind(max_tokens=max_tokens) if max_tokens else self._llm
        input_text = str([m.content for m in messages])
        with _langfuse.start_as_current_observation(name="achat", input=input_text) as obs:
            response = await llm.ainvoke(messages, config=_lf_config)
            content = response.content or ""
            obs.update(output=content)
        return content


def get_openai_chat_client() -> AzureChatClient:
    global _chat_client_instance
    if _chat_client_instance is None:
        with _chat_lock:
            if _chat_client_instance is None:
                _chat_client_instance = AzureChatClient()
    return _chat_client_instance
