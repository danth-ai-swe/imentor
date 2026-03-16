from functools import lru_cache
from typing import List, Dict

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.app_config import get_app_config, _SingletonMeta

config = get_app_config()
set_llm_cache(InMemoryCache())


def _retry_policy():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )


class AzureChatClient(metaclass=_SingletonMeta):

    @staticmethod
    @lru_cache(maxsize=1)
    def get_model() -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_deployment=config.OPENAI_CHAT_MODEL,
            api_version=config.OPENAI_API_VERSION,
            azure_endpoint=config.OPENAI_API_BASE,
            temperature=config.GPT_TEMPERATURE,
            top_p=config.GPT_TOP_P,
            max_tokens=config.GPT_MAX_TOKENS,
            timeout=config.GPT_TIMEOUT,
            api_key=config.OPENAI_API_KEY,
            max_retries=config.GPT_MAX_RETRIES,
        )

    @_retry_policy()
    def invoke(self, prompt: str) -> str:
        llm = self.get_model()
        response = llm.invoke(prompt)
        return response.content

    @_retry_policy()
    def invoke_full(self, prompt: str):
        llm = self.get_model()
        return llm.invoke(prompt)

    @_retry_policy()
    async def ainvoke(self, prompt: str) -> str:
        llm = self.get_model()
        response = await llm.ainvoke(prompt)
        return response.content

    @_retry_policy()
    async def ainvoke_full(self, prompt: str):
        llm = self.get_model()
        return await llm.ainvoke(prompt)

    @_retry_policy()
    def create_agentic_chunker_message(
            self,
            system_prompt: str,
            messages: List[Dict[str, str]],
    ) -> str:
        role_map = {
            "user": HumanMessage,
            "assistant": AIMessage,
        }

        lc_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            message_cls = role_map.get(role, HumanMessage)
            lc_messages.append(message_cls(content=content))

        llm = self.get_model()
        response = llm.invoke(lc_messages)
        return response.content


@lru_cache(maxsize=1)
def get_openai_chat_client() -> AzureChatClient:
    return AzureChatClient()
