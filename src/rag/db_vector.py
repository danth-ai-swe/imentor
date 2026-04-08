import asyncio
import uuid
from typing import Any, Optional

from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, AsyncQdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config.app_config import get_app_config
from src.constants.app_constant import (
    BM25_MODEL,
    BM25_OPTIONS,
    COLLECTION_NAME,
    DENSE_EMBEDDING_DIM,
)
from src.rag.llm.embedding_llm import get_openai_embedding_client
from src.utils.logger_utils import alog_method_call

config = get_app_config()


class QdrantManager:
    def __init__(
            self,
            collection_name: str,
            url: str = config.QDRANT_URL,
            colbert_model: str = "colbert-ir/colbertv2.0",
    ) -> None:
        self.collection_name = collection_name
        api_key = config.QDRANT_APIKEY if config.PROFILE_NAME == "prod" else None
        self.client = QdrantClient(url=url, api_key=api_key)
        self.async_client = AsyncQdrantClient(url=url, api_key=api_key)
        self.dense_embedder = get_openai_embedding_client()
        self._colbert = LateInteractionTextEmbedding(colbert_model)

    def _embed_colbert_doc(self, text: str) -> list[list[float]]:
        return list(self._colbert.embed([text]))[0].tolist()

    def _embed_colbert_query(self, query: str) -> list[list[float]]:
        return list(self._colbert.query_embed(query))[0].tolist()

    def _colbert_dim(self) -> int:
        sample = list(self._colbert.embed(["ping"]))[0]
        return len(sample[0])

    def create_collection(self, recreate: bool = False) -> None:
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        if self.client.collection_exists(self.collection_name):
            print(f"ℹ️  Collection '{self.collection_name}' already exists – skipped.")
            return

        late_dim = self._colbert_dim()

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_EMBEDDING_DIM,
                    distance=Distance.COSINE,
                    datatype=models.Datatype.FLOAT16,
                ),
                "maxsim": models.VectorParams(
                    size=late_dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    index=models.SparseIndexParams(datatype=models.Datatype.FLOAT16),
                ),
            },
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20_000,
                default_segment_number=5,
                max_segment_size=5_000_000,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=10,
                on_disk=False,
            ),
            strict_mode_config=models.StrictModeConfig(
                enabled=True,
                max_timeout=10,
                upsert_max_batchsize=1_000,
                read_rate_limit=5_000,
                write_rate_limit=5_000,
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True,
                    quantile=0.99,
                ),
            ),
        )
        print(f"✅ Collection '{self.collection_name}' created.")

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        print(f"🗑️  Collection '{self.collection_name}' deleted.")

    def create_payload_index(
            self,
            field_name: str,
            field_schema: models.PayloadSchemaType,
    ) -> None:
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
        print(f"✅ Payload index created on field '{field_name}'.")

    def upload_documents(
            self,
            documents: list[dict[str, Any]],
            batch_size: int = 256,
            parallel: int = 4,
            max_retries: int = 3,
    ) -> None:
        points: list[PointStruct] = []
        for doc in documents:
            text: str = doc["text"]
            late_vec = self._embed_colbert_doc(text)

            points.append(
                PointStruct(
                    id=doc.get("id", str(uuid.uuid4())),
                    vector={
                        "dense": self.dense_embedder.embed_query(text),
                        "maxsim": late_vec,
                        "sparse": models.Document(
                            text=text,
                            model=BM25_MODEL,
                            options=BM25_OPTIONS,
                        ),
                    },
                    payload={**doc.get("payload", {}), "text": text},
                )
            )

        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            update_mode=models.UpdateMode.INSERT_ONLY,
            parallel=parallel,
            max_retries=max_retries,
            batch_size=batch_size,
        )
        print(f"✅ {len(points)} document(s) uploaded.")

    def delete_by_filter(self, filter_conditions: list[models.Condition]) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=filter_conditions)
            ),
        )
        print("🗑️  Deleted points matching filter.")

    def get_by_ids(
            self,
            ids: list[int | str],
            with_vectors: bool = False,
            with_payload: bool = True,
    ) -> list:
        return self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_vectors=with_vectors,
            with_payload=with_payload,
        )

    def scroll(
            self,
            filter_conditions: Optional[list[models.Condition]] = None,
            limit: int = 15,
            order_by: Optional[str] = None,
            direction: models.Direction = models.Direction.DESC,
            with_payload: bool = True,
            with_vectors: bool = False,
    ) -> tuple[list, Any]:
        scroll_filter = (
            models.Filter(must=filter_conditions) if filter_conditions else None
        )
        results, next_offset = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            order_by=order_by,
            direction=direction,
        )
        return results, next_offset

    @staticmethod
    def build_filter(
            must: Optional[list[models.Condition]] = None,
            should: Optional[list[models.Condition]] = None,
            must_not: Optional[list[models.Condition]] = None,
            min_should: Optional[Any] = None,
    ) -> models.Filter:
        return models.Filter(
            must=must,
            should=should,
            must_not=must_not,
            min_should=min_should,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Async methods
    # ══════════════════════════════════════════════════════════════════════════

    async def _aembed_colbert_doc(self, text: str) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_colbert_doc, text)

    async def _aembed_colbert_query(self, query: str) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_colbert_query, query)

    @alog_method_call
    async def acreate_collection(self, recreate: bool = False) -> None:
        if recreate and await self.async_client.collection_exists(self.collection_name):
            await self.async_client.delete_collection(self.collection_name)

        if await self.async_client.collection_exists(self.collection_name):
            print(f"ℹ️  Collection '{self.collection_name}' already exists – skipped.")
            return

        late_dim = self._colbert_dim()

        await self.async_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=DENSE_EMBEDDING_DIM,
                    distance=Distance.COSINE,
                    datatype=models.Datatype.FLOAT16,
                ),
                "maxsim": models.VectorParams(
                    size=late_dim,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    index=models.SparseIndexParams(datatype=models.Datatype.FLOAT16),
                ),
            },
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20_000,
                default_segment_number=5,
                max_segment_size=5_000_000,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=10,
                on_disk=False,
            ),
            strict_mode_config=models.StrictModeConfig(
                enabled=True,
                max_timeout=10,
                upsert_max_batchsize=1_000,
                read_rate_limit=5_000,
                write_rate_limit=5_000,
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True,
                    quantile=0.99,
                ),
            ),
        )
        print(f"✅ Collection '{self.collection_name}' created.")

    @alog_method_call
    async def adelete_collection(self) -> None:
        await self.async_client.delete_collection(self.collection_name)
        print(f"🗑️  Collection '{self.collection_name}' deleted.")

    @alog_method_call
    async def acreate_payload_index(
            self,
            field_name: str,
            field_schema: models.PayloadSchemaType,
    ) -> None:
        await self.async_client.create_payload_index(
            collection_name=self.collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )
        print(f"✅ Payload index created on field '{field_name}'.")

    @alog_method_call
    async def aupload_documents(
            self,
            documents: list[dict[str, Any]],
            batch_size: int = 256,
            parallel: int = 4,
            max_retries: int = 3,
    ) -> None:
        points: list[PointStruct] = []
        for doc in documents:
            text: str = doc["text"]
            dense_vec, late_vec = await asyncio.gather(
                self.dense_embedder.aembed_query(text),
                self._aembed_colbert_doc(text),
            )

            points.append(
                PointStruct(
                    id=doc.get("id", str(uuid.uuid4())),
                    vector={
                        "dense": dense_vec,
                        "maxsim": late_vec,
                        "sparse": models.Document(
                            text=text,
                            model=BM25_MODEL,
                            options=BM25_OPTIONS,
                        ),
                    },
                    payload={**doc.get("payload", {}), "text": text},
                )
            )

        self.async_client.upload_points(
            collection_name=self.collection_name,
            points=points,
            parallel=parallel,
            max_retries=max_retries,
            batch_size=batch_size,
        )
        print(f"✅ {len(points)} document(s) uploaded.")

    @alog_method_call
    async def adelete_by_filter(self, filter_conditions: list[models.Condition]) -> None:
        await self.async_client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=filter_conditions)
            ),
        )
        print("🗑️  Deleted points matching filter.")

    @alog_method_call
    async def aget_by_ids(
            self,
            ids: list[int | str],
            with_vectors: bool = False,
            with_payload: bool = True,
    ) -> list:
        return await self.async_client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_vectors=with_vectors,
            with_payload=with_payload,
        )

    @alog_method_call
    async def ascroll(
            self,
            filter_conditions: Optional[list[models.Condition]] = None,
            limit: int = 15,
            order_by: Optional[str] = None,
            direction: models.Direction = models.Direction.DESC,
            with_payload: bool = True,
            with_vectors: bool = False,
    ) -> tuple[list, Any]:
        scroll_filter = (
            models.Filter(must=filter_conditions) if filter_conditions else None
        )
        results, next_offset = await self.async_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
            order_by=order_by,
            direction=direction,
        )
        return results, next_offset

    @alog_method_call
    async def ahybrid_search(
            self,
            query: str,
            top_k: int = 5,
            prefetch_limit: int = 10,
            rrf_k: int = 60,
            filter_conditions: Optional[list[models.Condition]] = None,
            with_payload: bool = True,
    ) -> list:
        dense_query_vec, colbert_query_vec = await asyncio.gather(
            self.dense_embedder.aembed_query(query),
            self._aembed_colbert_query(query),
        )
        query_filter = (
            models.Filter(must=filter_conditions) if filter_conditions else None
        )

        results = await self.async_client.query_points(
            collection_name=self.collection_name,
            query_filter=query_filter,
            with_vectors=False,
            with_payload=with_payload,
            limit=top_k,
            offset=0,
            using="maxsim",
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore=True),
                hnsw_ef=128,
                exact=False,
            ),
            prefetch=[
                models.Prefetch(
                    prefetch=[
                        models.Prefetch(
                            query=models.Document(
                                text=query,
                                model=BM25_MODEL,
                                options=BM25_OPTIONS,
                            ),
                            using="sparse",
                            params=models.SearchParams(
                                quantization=models.QuantizationSearchParams(rescore=False),
                            ),
                            limit=prefetch_limit,
                        ),
                        models.Prefetch(
                            query=dense_query_vec,
                            using="dense",
                            params=models.SearchParams(
                                quantization=models.QuantizationSearchParams(rescore=False),
                            ),
                            limit=prefetch_limit,
                        ),
                    ],
                    query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)),
                )
            ],
            query=colbert_query_vec,
        )
        return results.points

    @alog_method_call
    async def afacet(
            self,
            key: str,
            filter_conditions: Optional[list[models.Condition]] = None,
            exact: bool = True,
            limit: int = 10,
    ):
        facet_filter = (
            models.Filter(must=filter_conditions) if filter_conditions else None
        )
        return await self.async_client.facet(
            collection_name=self.collection_name,
            key=key,
            exact=exact,
            limit=limit,
            facet_filter=facet_filter,
        )

    @alog_method_call
    async def aget_collection_info(self):
        return await self.async_client.get_collection(self.collection_name)

    @alog_method_call
    async def aget_all_collections(self):
        return await self.async_client.get_collections()

    @alog_method_call
    async def aquery_points_groups(
            self,
            query: Any,
            group_by: str,
            limit: int = 2,
            group_size: int = 2,
            with_lookup: Optional[models.WithLookup] = None,
            filter_conditions: Optional[list[models.Condition]] = None,
            using: Optional[str] = None,
            with_payload: Any = True,
            with_vectors: Any = False,
    ):
        query_filter = (
            models.Filter(must=filter_conditions) if filter_conditions else None
        )
        return await self.async_client.query_points_groups(
            collection_name=self.collection_name,
            query=query,
            group_by=group_by,
            limit=limit,
            group_size=group_size,
            query_filter=query_filter,
            using=using,
            with_payload=with_payload,
            with_vectors=with_vectors,
            with_lookup=with_lookup,
        )

    @alog_method_call
    async def aquery_batch_points(
            self,
            requests: list[models.QueryRequest],
            consistency: Optional[Any] = None,
            timeout: Optional[int] = None,
    ):
        return await self.async_client.query_batch_points(
            collection_name=self.collection_name,
            requests=requests,
            consistency=consistency,
            timeout=timeout,
        )


_client_instance: "QdrantManager | None" = None


def get_qdrant_client() -> QdrantManager:
    global _client_instance
    if _client_instance is None:
        _client_instance = QdrantManager(collection_name=COLLECTION_NAME)
    return _client_instance
