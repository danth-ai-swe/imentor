from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.rag.llm.embedding_llm import get_openai_embedding_client

# ── Models & client ──────────────────────────────────────────────────────────
late_interaction_embedding_model = LateInteractionTextEmbedding("jinaai/jina-colbert-v2")
client = QdrantClient(url="http://localhost:6333")
collection_name = "test_collection2"

# ── Embed query ───────────────────────────────────────────────────────────────
query = "test"
late_vectors = list(late_interaction_embedding_model.query_embed(query))  # list of token-level vecs
dense_vector = get_openai_embedding_client().embed_query(query)  # 1536-dim float list

# ── Create collection (once) ──────────────────────────────────────────────────
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=1536,
                distance=Distance.COSINE,
                datatype=models.Datatype.FLOAT16,
            ),
            "maxsim": models.VectorParams(
                size=len(late_vectors[0]),  # FIX: was late_interaction_embeddings[0][0]
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),  # Disable HNSW for reranking
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
                index=models.SparseIndexParams(datatype=models.Datatype.FLOAT16),
            ),
        },
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=20000,
            default_segment_number=5,
            max_segment_size=5_000_000,
        ),
        hnsw_config=models.HnswConfigDiff(m=6, on_disk=False),
        strict_mode_config=models.StrictModeConfig(
            enabled=True,
            max_timeout=10,
            upsert_max_batchsize=1000,
            read_rate_limit=5000,
            write_rate_limit=5000,
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
                quantile=0.99,
            ),
        ),
    )

    # ── Upsert (sequential, with order guarantee) ─────────────────────────────
    # FIX: multi-vector collection requires a dict of named vectors per point
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[
            PointStruct(
                id=1,
                vector={"dense": dense_vector},  # FIX: named vector dict
                payload={"city": "Berlin"},
            ),
            PointStruct(
                id=2,
                vector={
                    "dense": dense_vector,
                    "sparse": models.Document(  # server-side BM25
                        text="Recipe for baking chocolate chip cookies",
                        model="Qdrant/bm25",
                        options={
                            "language": "none",
                            "ascii_folding": True,
                            "tokenizer": "multilingual",
                        },
                    ),
                },
                payload={"city": "London"},
            ),
        ],
        update_mode=models.UpdateMode.INSERT_ONLY,
    )
    print("upsert result:", operation_info)

    # ── Upsert (parallel batch) ───────────────────────────────────────────────
    client.upload_points(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=3,
                vector={"dense": dense_vector},
                payload={"city": "Berlin"},
            ),
            PointStruct(
                id=4,
                vector={"dense": dense_vector},
                payload={"city": "London"},
            ),
        ],
        update_mode=models.UpdateMode.INSERT_ONLY,
        parallel=4,
        max_retries=3,
        batch_size=256,
    )

    # ── Delete by id ──────────────────────────────────────────────────────────
    client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(points=[100]),
    )

    # ── Delete by filter ──────────────────────────────────────────────────────
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="color",
                        match=models.MatchValue(value="red"),
                    )
                ]
            )
        ),
    )

    # ── Retrieve by id ────────────────────────────────────────────────────────
    results = client.retrieve(
        collection_name=collection_name,
        ids=[1, 2],
    )
    print("retrieve:", results)

    # ── Scroll (filter + sort) ────────────────────────────────────────────────
    scroll_result, _next = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="city",
                    match=models.MatchValue(value="London"),
                )
            ]
        ),
        limit=15,
        with_payload=True,
        with_vectors=False,
    )
    print("scroll:", scroll_result)

# ── Hybrid search (prefetch dense + sparse → rerank with ColBERT maxsim) ────
search_result = client.query_points(
    collection_name=collection_name,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="city",
                match=models.MatchValue(value="London"),
            )
        ]
    ),
    with_vectors=False,
    with_payload=True,
    limit=5,
    using="maxsim",  # FIX: "parse" → "maxsim" (must match collection)
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(rescore=True),
        hnsw_ef=128,
        exact=False,
    ),
    prefetch=[
        # Stage 1a – sparse BM25 retrieval
        models.Prefetch(
            query=models.Document(
                text=query,
                model="Qdrant/bm25",
                options={
                    "language": "none",
                    "ascii_folding": True,
                    "tokenizer": "multilingual",
                },
            ),
            using="sparse",
            params=models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore=False),
            ),
            limit=50,
        ),
        # Stage 1b – dense ANN retrieval
        models.Prefetch(
            query=dense_vector,
            using="dense",
            params=models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore=False),
            ),
            limit=50,
        ),
    ],
    # Stage 2 – ColBERT late-interaction rerank over prefetch candidates
    query=models.Document(text=query, model="jinaai/jina-colbert-v2"),
)
print("hybrid search:", search_result)

# ── Multi-stage search (byte MRL → full dense → ColBERT) ─────────────────────
multistage_result = client.query_points(
    collection_name=collection_name,
    prefetch=models.Prefetch(
        prefetch=models.Prefetch(
            query=[1, 23, 45, 67],  # small byte vector
            using="mrl_byte",
            limit=1000,
        ),
        query=[0.01, 0.45, 0.67],  # full dense vector
        using="full",
        limit=100,
    ),
    query=[
        [0.17, 0.23, 0.52],
        [0.22, 0.11, 0.63],
        [0.86, 0.93, 0.12],
    ],
    using="colbert",
    limit=10,
)
print("multi-stage:", multistage_result)

# ── Group search with lookup ──────────────────────────────────────────────────
group_result = client.query_points_groups(
    collection_name="chunks",
    query=[1.1],
    group_by="document_id",
    limit=2,
    group_size=2,
    with_lookup=models.WithLookup(
        collection="documents",
        with_payload=["title", "text"],
        with_vectors=False,
    ),
)
print("groups:", group_result)

# ── Facet ─────────────────────────────────────────────────────────────────────
facet_result = client.facet(
    collection_name=collection_name,
    key="size",
    exact=True,
    facet_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="color",
                match=models.MatchValue(value="red"),
            )
        ]
    ),
)
print("facet:", facet_result)

# ── Collection metadata ───────────────────────────────────────────────────────
print(client.get_collection_aliases(collection_name=collection_name))
print(client.get_aliases())
print(client.get_collections())
