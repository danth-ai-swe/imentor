from functools import lru_cache

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_openai import AzureOpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.app_config import get_app_config, _SingletonMeta

set_llm_cache(InMemoryCache())

config = get_app_config()


def _retry_policy():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )


class AzureEmbeddingClient(metaclass=_SingletonMeta):

    @staticmethod
    @lru_cache(maxsize=1)
    def get_model() -> AzureOpenAIEmbeddings:
        return AzureOpenAIEmbeddings(
            azure_deployment=config.OPENAI_EMBEDDING_MODEL,
            api_version=config.OPENAI_API_VERSION,
            azure_endpoint=config.OPENAI_API_BASE,
            timeout=config.GPT_TIMEOUT,
            api_key=config.OPENAI_API_KEY,
            max_retries=config.GPT_MAX_RETRIES
        )

    @_retry_policy()
    def embed_query(self, text: str) -> list[float]:
        model = self.get_model()
        return model.embed_query(text)

    @_retry_policy()
    def embed_query_full(self, text: str):
        model = self.get_model()
        client = model.client
        response = client.create(
            input=[text],
            model=model.deployment,
            encoding_format="float",
        )
        return response

    @_retry_policy()
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self.get_model()
        return model.embed_documents(texts)

    @_retry_policy()
    async def aembed_query(self, text: str) -> list[float]:
        model = self.get_model()
        return await model.aembed_query(text)

    @_retry_policy()
    async def aembed_query_full(self, text: str):
        model = self.get_model()
        async_client = model.async_client
        response = await async_client.create(
            input=[text],
            model=model.deployment,
            encoding_format="float",
        )
        return response

    @_retry_policy()
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self.get_model()
        return await model.aembed_documents(texts)


@lru_cache(maxsize=1)
def get_openai_embedding_client() -> AzureEmbeddingClient:
    return AzureEmbeddingClient()
