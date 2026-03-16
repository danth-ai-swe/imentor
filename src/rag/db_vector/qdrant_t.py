import uuid

from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.rag.llm.embedding_llm import get_openai_embedding_client

# ─────────────────────────────────────────────
# INIT MODELS & CLIENT
# ─────────────────────────────────────────────

late_interaction_embedding_model = LateInteractionTextEmbedding("jinaai/jina-colbert-v2")
# Sparse BM25 được xử lý server-side qua models.Document — không cần SparseTextEmbedding local

client = QdrantClient(url="http://localhost:6333")  # client = qdrant_client.AsyncQdrantClient("localhost")
collection_name = "test_collection23"

# ─────────────────────────────────────────────
# LOAD 1 TÀI LIỆU CƠ BẢN
# ─────────────────────────────────────────────

# Đọc nội dung tài liệu (thay bằng đường dẫn file thực tế của bạn)
DOCUMENT_PATH = "sample_document.txt"

try:
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
        document_text = f.read()
    print(f"✅ Loaded document: {DOCUMENT_PATH} ({len(document_text)} chars)")
except FileNotFoundError:
    # Dùng nội dung mẫu nếu không tìm thấy file
    document_text = (
        "Qdrant is a vector similarity search engine and database. "
        "It provides a production-ready service with a convenient API to store, search, and manage vectors "
        "with an additional payload. Qdrant supports dense, sparse, and late-interaction (ColBERT) vectors, "
        "enabling powerful hybrid search pipelines."
    )
    print(f"⚠️  File not found — using sample text ({len(document_text)} chars)")

document_id = str(uuid.uuid4())  # ID duy nhất cho tài liệu

# ─────────────────────────────────────────────
# EMBED DOCUMENT
# ─────────────────────────────────────────────

query = "What is Qdrant?"

# Dense vector (OpenAI text-embedding-3-small, dim=1536)
openai_embeddings = get_openai_embedding_client()
dense_vector_doc = openai_embeddings.embed_query(document_text)
dense_vector_query = openai_embeddings.embed_query(query)

# Late-interaction / ColBERT vectors (list of token vectors)
late_vectors_doc = list(late_interaction_embedding_model.embed([document_text]))[0]  # (tokens, dim)
late_vectors_query = list(late_interaction_embedding_model.query_embed(query))[0]  # (tokens, dim)
late_dim = len(late_vectors_doc[0])

# Sparse BM25 — dùng models.Document, Qdrant server tự embed server-side
# Không cần tạo vector thủ công ở client
sparse_doc = models.Document(
    text=document_text,
    model="Qdrant/bm25",
    options={"language": "none", "ascii_folding": True, "tokenizer": "multilingual"},
)
sparse_query = models.Document(
    text=query,
    model="Qdrant/bm25",
    options={"language": "none", "ascii_folding": True, "tokenizer": "multilingual"},
)

print(f"Dense dim       : {len(dense_vector_doc)}")
print(f"Late-interaction: tokens={len(late_vectors_doc)}, dim_per_token={late_dim}")

# ─────────────────────────────────────────────
# CREATE COLLECTION (nếu chưa tồn tại)
# ─────────────────────────────────────────────

if not client.collection_exists(collection_name=collection_name):
    # client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=1536,
                distance=Distance.COSINE,
                datatype=models.Datatype.FLOAT16,
            ),
            # , on_disk=True
            "maxsim": models.VectorParams(
                size=late_dim,
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
            max_segment_size=5000000,
        ),
        hnsw_config=models.HnswConfigDiff(
            m=6,  # decrease M for lower memory usage
            on_disk=False,
        ),
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
    print(f"✅ Collection '{collection_name}' created.")

    # client.create_payload_index(
    #     collection_name="{collection_name}",
    #     field_name="name_of_the_field_to_index",
    #     field_schema=models.TextIndexParams(type=models.TextIndexType.TEXT, ascii_folding=True),
    # )
    # Available field types are:
    #
    # keyword - for keyword payload, affects Match filtering conditions.
    # integer - for integer payload, affects Match and Range filtering conditions.
    # float   - for float payload, affects Range filtering conditions.
    # bool    - for bool payload, affects Match filtering conditions (available as of v1.4.0).
    # geo     - for geo payload, affects Geo Bounding Box and Geo Radius filtering conditions.
    # datetime- for datetime payload, affects Range filtering conditions (available as of v1.8.0).
    # text    - a special kind of index, available for keyword / string payloads, affects Full Text search filtering conditions.
    # uuid    - a special type of index, similar to keyword, but optimized for UUID values. (available as of v1.11.0)

    # ─────────────────────────────────────────
    # INSERT DOCUMENT (insert parallel)
    # ─────────────────────────────────────────

    client.upload_points(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=1,
                vector={
                    "dense": dense_vector_doc,
                    "maxsim": late_vectors_doc.tolist(),
                    # sparse dùng models.Document — Qdrant server tự tokenize & embed BM25
                    "sparse": sparse_doc,
                },
                payload={
                    "document_id": document_id,
                    "text": document_text,
                    "city": "Berlin",
                    "source": DOCUMENT_PATH,
                },
            ),
        ],
        update_mode=models.UpdateMode.INSERT_ONLY,
        parallel=4,
        max_retries=3,
        batch_size=256,  # How many vectors will be uploaded in a single request?
    )
    print("✅ Document inserted.")

    # delete by id
    client.delete(
        collection_name=collection_name,
        points_selector=models.PointIdsList(
            points=[0, 3, 100],
        ),
    )
    # delete by filter
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="color",
                        match=models.MatchValue(value="red"),
                    ),
                ],
            )
        ),
    )
    # get by id
    retrieved = client.retrieve(
        collection_name=collection_name,
        ids=[1],
    )
    print(f"Retrieved by id: {retrieved}")

    # get all
    scroll_result, next_offset = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="city", match=models.MatchValue(value="Berlin")),
            ]
        ),
        limit=15,
        with_payload=True,
        with_vectors=False,
        order_by="document_id",  # sort; When you use the order_by parameter, pagination is disabled.
        direction=models.Direction.DESC,  # default is "ASC"
    )
    print(f"Scroll results: {scroll_result}")

    # ─────────────────────────────────────────
    # HYBRID SEARCH
    # ─────────────────────────────────────────

    search_results = client.query_points(
        collection_name=collection_name,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="city",
                    match=models.MatchValue(value="Berlin"),
                )
            ]
        ),
        with_vectors=False,
        with_payload=True,
        # with_payload=["city", "village", "town"],
        # | models.PayloadSelectorExclude(exclude=["city"]),
        limit=5,  # Max amount of results
        offset=0,  # Large offset values may cause performance issues
        using="maxsim",
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                rescore=True,
            ),
            hnsw_ef=128,
            exact=False,
        ),
        prefetch=[
            models.Prefetch(
                prefetch=[
                    models.Prefetch(
                        # sparse dùng models.Document — Qdrant server tự tokenize & embed BM25
                        query=models.Document(
                            text=query,
                            model="Qdrant/bm25",
                            options={"language": "none", "ascii_folding": True, "tokenizer": "multilingual"},
                        ),
                        params=models.SearchParams(
                            quantization=models.QuantizationSearchParams(
                                # Avoid rescoring in prefetch
                                # We should do it explicitly on the second stage
                                rescore=False,
                            ),
                        ),
                        using="sparse",
                        limit=50,
                    ),
                    models.Prefetch(
                        query=dense_vector_query,  # <-- dense vector
                        using="dense",
                        params=models.SearchParams(
                            quantization=models.QuantizationSearchParams(
                                # Avoid rescoring in prefetch
                                # We should do it explicitly on the second stage
                                rescore=False,
                            ),
                        ),
                        limit=50,
                    ),
                ],
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF,
                    rrf=models.Rrf(weights=[3.0, 1.0]),
                ),
            )
        ],
        query=late_vectors_query.tolist(),  # ColBERT MaxSim reranking on fused results
    )
    print(f"\n🔍 Hybrid search results for: '{query}'")
    for point in search_results.points:
        print(f"  id={point.id}  score={point.score:.4f}  payload={point.payload}")

    # ─────────────────────────────────────────
    # FACET (group by)
    # ─────────────────────────────────────────

    facet_result = client.facet(
        collection_name=collection_name,
        key="city",
        exact=True,
        facet_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="city",
                    match=models.MatchValue(value="Berlin"),
                )
            ]
        ),
    )
    print(f"\nFacet result: {facet_result}")

else:
    print(f"ℹ️  Collection '{collection_name}' already exists — skipping creation & insertion.")

# ─────────────────────────────────────────────
# COLLECTION INFO
# ─────────────────────────────────────────────

print(client.get_collection(collection_name=collection_name))
print(client.get_collection_aliases(collection_name=collection_name))
print(client.get_aliases())
print(client.get_collections())
