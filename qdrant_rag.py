import uuid
from dataclasses import dataclass, field
from typing import Any

from src.rag.db_vector.qdrant_client_t import AsyncQdrantClient, QdrantClient
from src.rag.db_vector.qdrant_client_t import models as qmodels
from src.rag.db_vector.qdrant_client_t import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)


@dataclass
class QdrantConfig:
    """
    All knobs for connecting to Qdrant and defining a collection.

    Local (in-memory or on-disk):
        host = None, path = ":memory:" | "/data/qdrant"

    Remote (cloud or self-hosted):
        host = "localhost" | "xyz.cloud.qdrant.io"
        port = 6333
        api_key = "<your-key>"         # required for Qdrant Cloud
        https = True                   # required for Qdrant Cloud
    """

    # --- Connection --------------------------------------------------------
    host: str | None = "localhost"  # None → use `path` for local mode
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False  # True → faster bulk upserts
    https: bool = False
    api_key: str | None = None
    timeout: int = 30
    path: str | None = None  # local persistent path (no host needed)

    # --- Collection --------------------------------------------------------
    collection_name: str = "rag_documents"
    vector_size: int = 1536  # OpenAI ada-002 / text-embedding-3-small
    distance: Distance = Distance.COSINE

    # HNSW index parameters (tune for recall vs speed)
    hnsw_m: int = 16  # number of edges per node
    hnsw_ef_construct: int = 100  # build-time search width

    # Quantization — set to True to enable scalar int8 quantization (saves ~4x RAM)
    enable_scalar_quantization: bool = False

    # Payload (metadata) indexing — list of field names to index for fast filtering
    payload_index_fields: list[str] = field(default_factory=list)

    # Sparse vectors for hybrid search (requires Qdrant ≥ 1.7)
    enable_sparse_vectors: bool = False
    sparse_vector_name: str = "bm25"


# ---------------------------------------------------------------------------
# Document / chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single unit of text to be embedded and stored."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Core Qdrant RAG class
# ---------------------------------------------------------------------------

class QdrantRAG:
    """
    Drop-in Qdrant backend for RAG pipelines.

    Usage (sync):
        cfg = QdrantConfig(host="localhost", collection_name="docs", vector_size=1536)
        rag = QdrantRAG(cfg)
        rag.ensure_collection()

        docs = [Document(text="Hello world", metadata={"source": "manual"})]
        rag.upsert_documents(docs, embeddings=[[0.1, ...]])

        results = rag.search("What is hello?", query_embedding=[0.1, ...], top_k=5)

    Usage (async):
        async with QdrantRAG.async_client(cfg) as rag:
            await rag.async_upsert_documents(docs, embeddings=embeddings)
            results = await rag.async_search(query_embedding=vec, top_k=5)
    """

    def __init__(self, _config: QdrantConfig) -> None:
        self.config = _config
        self._client: QdrantClient = self._build_client()
        self._async_client: AsyncQdrantClient | None = None

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------

    def _build_client(self) -> QdrantClient:
        cfg = self.config
        if cfg.host is None and cfg.path is not None:
            # Local file-based or in-memory Qdrant
            return QdrantClient(path=cfg.path, timeout=cfg.timeout)
        return QdrantClient(
            host=cfg.host,
            port=cfg.port,
            grpc_port=cfg.grpc_port,
            prefer_grpc=cfg.prefer_grpc,
            https=cfg.https,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
        )

    def _build_async_client(self) -> AsyncQdrantClient:
        cfg = self.config
        if cfg.host is None and cfg.path is not None:
            return AsyncQdrantClient(path=cfg.path)
        return AsyncQdrantClient(
            host=cfg.host,
            port=cfg.port,
            grpc_port=cfg.grpc_port,
            prefer_grpc=cfg.prefer_grpc,
            https=cfg.https,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
        )

    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def async_client(self) -> AsyncQdrantClient:
        if self._async_client is None:
            self._async_client = self._build_async_client()
        return self._async_client

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        return self._client.collection_exists(self.config.collection_name)

    def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if it doesn't exist. Pass recreate=True to drop and rebuild."""
        name = self.config.collection_name
        cfg = self.config

        if recreate and self.collection_exists():
            self._client.delete_collection(name)

        if not self.collection_exists():
            vectors_config: dict[str, VectorParams] | VectorParams

            if cfg.enable_sparse_vectors:
                # Named dense vector + sparse BM25 vector
                vectors_config = {
                    "dense": VectorParams(
                        size=cfg.vector_size,
                        distance=cfg.distance,
                        hnsw_config=qmodels.HnswConfigDiff(
                            m=cfg.hnsw_m,
                            ef_construct=cfg.hnsw_ef_construct,
                        ),
                    )
                }
                sparse_vectors_config = {
                    cfg.sparse_vector_name: qmodels.SparseVectorParams(
                        index=qmodels.SparseIndexParams(on_disk=False)
                    )
                }
            else:
                vectors_config = VectorParams(
                    size=cfg.vector_size,
                    distance=cfg.distance,
                    hnsw_config=qmodels.HnswConfigDiff(
                        m=cfg.hnsw_m,
                        ef_construct=cfg.hnsw_ef_construct,
                    ),
                )
                sparse_vectors_config = None

            quantization_config = (
                qmodels.ScalarQuantizationConfig(
                    scalar=qmodels.ScalarQuantization(
                        type=qmodels.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                )
                if cfg.enable_scalar_quantization else None
            )

            self._client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                quantization_config=quantization_config,
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=20_000,  # delay indexing until batch is large
                ),
                on_disk_payload=True,  # keep payloads off RAM
            )

        # Create payload indexes for fast metadata filtering
        for field_name in cfg.payload_index_fields:
            self._client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

    def delete_collection(self) -> None:
        self._client.delete_collection(self.config.collection_name)

    def collection_info(self) -> dict[str, Any]:
        info = self._client.get_collection(self.config.collection_name)
        return {
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "optimizer_status": info.optimizer_status,
        }

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_documents(
            self,
            documents: list[Document],
            _embeddings: list[list[float]],
            batch_size: int = 256,
            sparse_embeddings: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Upsert documents with their pre-computed dense embeddings.

        Args:
            documents:         List of Document objects.
            _embeddings:        Dense embedding per document (same order).
            batch_size:        Points sent per request.
            sparse_embeddings: Optional list of {"indices": [...], "values": [...]}
                               dicts for hybrid search (one per document).
        """
        cfg = self.config
        points: list[PointStruct] = []

        for i, (doc, emb) in enumerate(zip(documents, _embeddings)):
            payload = {"text": doc.text, **doc.metadata}

            if cfg.enable_sparse_vectors and sparse_embeddings:
                sp = sparse_embeddings[i]
                vector = {
                    "dense": emb,
                    cfg.sparse_vector_name: qmodels.SparseVector(
                        indices=sp["indices"], values=sp["values"]
                    ),
                }
            else:
                vector = emb  # type: ignore[assignment]

            points.append(
                PointStruct(id=doc.doc_id, vector=vector, payload=payload)
            )

        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start: batch_start + batch_size]
            self._client.upsert(
                collection_name=cfg.collection_name,
                points=batch,
                wait=True,
            )

    # ------------------------------------------------------------------
    # Search — dense (semantic)
    # ------------------------------------------------------------------

    def search(
            self,
            query_embedding: list[float],
            top_k: int = 5,
            score_threshold: float | None = None,
            filters: dict[str, Any] | None = None,
            vector_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Dense (semantic) nearest-neighbour search.

        Args:
            query_embedding:  Query vector.
            top_k:            Maximum results to return.
            score_threshold:  Minimum similarity score (0-1 for cosine).
            filters:          Simple equality filters, e.g. {"source": "manual"}.
            vector_name:      Named vector to search (required when sparse is enabled).

        Returns:
            List of dicts with keys: id, score, text, metadata.
        """
        qdrant_filter = self._build_filter(filters) if filters else None
        search_vector = (
            (vector_name or "dense", query_embedding)
            if self.config.enable_sparse_vectors
            else query_embedding
        )

        hits: list[ScoredPoint] = self._client.search(
            collection_name=self.config.collection_name,
            query_vector=search_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
        )
        return [self._format_hit(h) for h in hits]

    # ------------------------------------------------------------------
    # Search — hybrid (dense + sparse fusion via Reciprocal Rank Fusion)
    # ------------------------------------------------------------------

    def hybrid_search(
            self,
            dense_embedding: list[float],
            sparse_indices: list[int],
            sparse_values: list[float],
            top_k: int = 5,
            filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining dense ANN and sparse BM25 via RRF fusion.
        Requires enable_sparse_vectors=True in config.
        """
        if not self.config.enable_sparse_vectors:
            raise RuntimeError("Hybrid search requires enable_sparse_vectors=True in QdrantConfig.")

        qdrant_filter = self._build_filter(filters) if filters else None

        _results = self._client.query_points(
            collection_name=self.config.collection_name,
            prefetch=[
                qmodels.Prefetch(
                    query=dense_embedding,
                    using="dense",
                    limit=top_k * 3,
                    filter=qdrant_filter,
                ),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(
                        indices=sparse_indices, values=sparse_values
                    ),
                    using=self.config.sparse_vector_name,
                    limit=top_k * 3,
                    filter=qdrant_filter,
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [self._format_hit(h) for h in _results.points]

    # ------------------------------------------------------------------
    # Scroll / full-scan retrieval
    # ------------------------------------------------------------------

    def scroll(
            self,
            filters: dict[str, Any] | None = None,
            limit: int = 100,
            offset: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """
        Paginate through all points (no scoring).

        Returns:
            (results, next_page_offset) — pass next_page_offset as `offset`
            for the following page; None means last page.
        """
        qdrant_filter = self._build_filter(filters) if filters else None
        records, next_offset = self._client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        return [self._format_record(_r) for _r in records], next_offset  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Delete points
    # ------------------------------------------------------------------

    def delete_by_ids(self, ids: list[str]) -> None:
        self._client.delete(
            collection_name=self.config.collection_name,
            points_selector=qmodels.PointIdsList(points=ids),
        )

    def delete_by_filter(self, filters: dict[str, Any]) -> None:
        self._client.delete(
            collection_name=self.config.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=self._build_filter(filters)
            ),
        )

    # ------------------------------------------------------------------
    # Async variants
    # ------------------------------------------------------------------

    async def async_upsert_documents(
            self,
            documents: list[Document],
            _embeddings: list[list[float]],
            batch_size: int = 256,
    ) -> None:
        cfg = self.config
        points = [
            PointStruct(
                id=doc.doc_id,
                vector=emb,
                payload={"text": doc.text, **doc.metadata},
            )
            for doc, emb in zip(documents, _embeddings)
        ]
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start: batch_start + batch_size]
            await self.async_client.upsert(
                collection_name=cfg.collection_name,
                points=batch,
                wait=True,
            )

    async def async_search(
            self,
            query_embedding: list[float],
            top_k: int = 5,
            score_threshold: float | None = None,
            filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        qdrant_filter = self._build_filter(filters) if filters else None
        hits = await self.async_client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
        )
        return [self._format_hit(h) for h in hits]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        """Convert a flat dict of {field: value} into a Qdrant Filter (AND logic)."""
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        return Filter(must=conditions)

    @staticmethod
    def _format_hit(hit: ScoredPoint) -> dict[str, Any]:
        payload = hit.payload or {}
        return {
            "id": hit.id,
            "score": hit.score,
            "text": payload.pop("text", ""),
            "metadata": payload,
        }

    @staticmethod
    def _format_record(record: qmodels.Record) -> dict[str, Any]:
        payload = dict(record.payload or {})
        return {
            "id": record.id,
            "score": None,
            "text": payload.pop("text", ""),
            "metadata": payload,
        }

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"QdrantRAG(collection={cfg.collection_name!r}, "
            f"host={cfg.host!r}, vector_size={cfg.vector_size})"
        )


# ---------------------------------------------------------------------------
# Chunking utility
# ---------------------------------------------------------------------------

def chunk_text(
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
) -> list[str]:
    """
    Split text into overlapping word-boundary chunks.

    Args:
        text:          Raw text to split.
        chunk_size:    Target chunk size in characters.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    _chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Walk back to a word boundary
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        _chunks.append(text[start:end].strip())
        start = end - chunk_overlap
    return [c for c in _chunks if c]


# ---------------------------------------------------------------------------
# Quick-start example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    VECTOR_SIZE = 128  # small for demo

    # 1. Configure
    config = QdrantConfig(
        host=None,  # in-memory local instance
        path=":memory:",
        collection_name="demo",
        vector_size=VECTOR_SIZE,
        distance=Distance.COSINE,
        payload_index_fields=["source"],
        enable_scalar_quantization=False,
        enable_sparse_vectors=False,
    )

    # 2. Initialise & create collection
    rag = QdrantRAG(config)
    rag.ensure_collection()
    print("Collection created:", rag.collection_info())

    # 3. Fake documents + embeddings
    raw_texts = [
        "Qdrant is a high-performance vector database.",
        "RAG combines retrieval with generation for grounded LLM answers.",
        "Embeddings map text into a high-dimensional semantic space.",
        "Cosine similarity measures the angle between two vectors.",
        "Chunking splits long documents into overlapping segments.",
    ]
    docs = [
        Document(text=t, metadata={"source": "demo", "index": i})
        for i, t in enumerate(raw_texts)
    ]
    # Random unit vectors — replace with real embeddings in production
    embeddings = [
        [random.gauss(0, 1) for _ in range(VECTOR_SIZE)] for _ in raw_texts
    ]

    # 4. Upsert
    rag.upsert_documents(docs, embeddings)
    print("Upserted", len(docs), "documents.")
    print("Collection info:", rag.collection_info())

    # 5. Search
    query_vec = [random.gauss(0, 1) for _ in range(VECTOR_SIZE)]
    results = rag.search(query_vec, top_k=3, filters={"source": "demo"})
    print("\nTop-3 search results:")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['text'][:60]}")

    # 6. Chunking demo
    long_text = "word " * 300
    chunks = chunk_text(long_text, chunk_size=100, chunk_overlap=20)
    print(f"\nChunked long text into {len(chunks)} chunks.")
