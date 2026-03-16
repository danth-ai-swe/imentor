from fastembed import LateInteractionTextEmbedding

from src.rag.db_vector.qdrant_client_t import Distance, VectorParams
from src.rag.db_vector.qdrant_client_t import QdrantClient, models
from src.rag.llm.embedding_llm import get_openai_embedding_client
from src.rag.db_vector.qdrant_client_t import PointStruct
late_interaction_embedding_model = LateInteractionTextEmbedding("jinaai/jina-colbert-v2")
client = QdrantClient(url="http://localhost:6333")  # client = qdrant_client.AsyncQdrantClient("localhost")
collection_name = "test_collection2"
query = "test"
late_vectors = late_interaction_embedding_model.query_embed(query)
vector = get_openai_embedding_client().embed_query("test")
if not client.collection_exists(collection_name=collection_name):
    # client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=1536, distance=Distance.COSINE, datatype=models.Datatype.FLOAT16),
            # , on_disk=True
            "maxsim": models.VectorParams(
                size=len(late_interaction_embeddings[0][0]),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0)  # Disable HNSW for reranking
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF,
                                                index=models.SparseIndexParams(datatype=models.Datatype.FLOAT16)),
        },
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000, default_segment_number=5,
                                                      max_segment_size=5000000),
        hnsw_config=models.HnswConfigDiff(m=6,  # decrease M for lower memory usage
                                          on_disk=False),
        strict_mode_config=models.StrictModeConfig(enabled=True, max_timeout=10, upsert_max_batchsize=1000,
                                                   read_rate_limit=5000, write_rate_limit=5000),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
                quantile=0.99,
            ),
        ),
    )
    # client.create_payload_index(
    #     collection_name="{collection_name}",
    #     field_name="name_of_the_field_to_index",
    #     field_schema=models.TextIndexParams(type=models.TextIndexType.TEXT, ascii_folding=True),
    # )
    # Available field types are:
    #
    # keyword - for keyword payload, affects Match filtering conditions.
    # integer - for integer payload, affects Match and Range filtering conditions.
    # float - for float payload, affects Range filtering conditions.
    # bool - for bool payload, affects Match filtering conditions (available as of v1.4.0).
    # geo - for geo payload, affects Geo Bounding Box and Geo Radius filtering conditions.
    # datetime - for datetime payload, affects Range filtering conditions (available as of v1.8.0).
    # text - a special kind of index, available for keyword / string payloads, affects Full Text search filtering conditions. Read more about text index configuration
    # uuid - a special type of index, similar to keyword, but optimized for UUID values. Affects Match filtering conditions. (available as of v1.11.0)


    # insert parallel
    client.upload_points(
        collection_name=collection_name,
        points=[
            PointStruct(id=1, vector=vector, payload={"city": "Berlin"}),
            PointStruct(id=2, vector={
                "vector": vector,
                "text": models.Document(
                    text="Recipe for baking chocolate chip cookies",
                    model="Qdrant/bm25",
                    options={"language": "none", "ascii_folding": True, "tokenizer": "multilingual"},
                )
            }, payload={"city": "London"}),
        ],
        update_mode=models.UpdateMode.INSERT_ONLY,
        parallel=4,
        max_retries=3,
        using="vector",
        ids=None,  # Vector ids will be assigned automatically
        batch_size=256,  # How many vectors will be uploaded in a single request?
    )
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
    client.retrieve(
        collection_name=collection_name,
        ids=[0, 3, 100],
    )
    # get all
    client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="color", match=models.MatchValue(value="red")),
            ]
        ),
        limit=15,
        with_payload=True,
        with_vectors=False,
        order_by="timestamp",  # sort ,When you use the order_by parameter, pagination is disabled.
        direction=models.Direction.DESC,  # default is "ASC"
        start_from=123,
    )

    #     hibrid search
    client.query_points(
        collection_name=collection_name,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="city",
                    match=models.MatchValue(
                        value="London",
                    ),
                )
            ]
        ),
        with_vectors=True,
        with_payload=True,
        # with_payload=["city", "village", "town"], | models.PayloadSelectorExclude(  exclude=["city"], ),
        group_by="document_id",  # Path of the field to group by
        limit=5,  # Max amount of groups
        group_size=2,  # Max amount of points per group
        offset=1,  # Large offset values may cause performance issues
        using="maxsim",

        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                rescore=True,
            ),
            acorn=models.AcornSearchParams(
                enable=True,
                max_selectivity=0.4,
            ),
            hnsw_ef=128, exact=False
        ),
        lookup_from=models.LookupLocation(
            collection="another_collection",  # <--- other collection name
            vector="image-512",  # <--- vector name in the other collection
        ),
        prefetch=[
            models.Prefetch(
                prefetch=[
                    models.Prefetch(
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
                        using="parse",
                        limit=50,
                    ),
                    models.Prefetch(
                        query=vector,  # <-- dense vector
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
                query=models.FusionQuery(fusion=models.Fusion.RRF, rrf=models.Rrf(weights=[3.0, 1.0])),
            )
        ],
        query=models.Document(text=query, model="jinaai/jina-colbert-v2"),
        # query=models.FormulaQuery(
        #     formula=models.SumExpression(sum=[
        #         "$score",
        #         models.MultExpression(
        #             mult=[0.5, models.FieldCondition(key="tag", match=models.MatchAny(any=["h1", "h2", "h3", "h4"]))]),
        #         models.MultExpression(
        #             mult=[0.25, models.FieldCondition(key="tag", match=models.MatchAny(any=["p", "li"]))])
        #     ]
        #     ))
    )

    client.query_points_groups(
        collection_name="chunks",
        # Same as in the regular search() API
        query=[1.1],
        # Grouping parameters
        group_by="document_id",  # Path of the field to group by
        limit=2,  # Max amount of groups
        group_size=2,  # Max amount of points per group
        # Lookup parameters
        with_lookup=models.WithLookup(
            # For the with_lookup parameter, you can also use the shorthand with_lookup="documents" to bring the whole payload and vector(s) without explicitly specifying it.
            # Name of the collection to look up points in
            collection="documents",
            # Options for specifying what to bring from the payload
            # of the looked up point, True by default
            with_payload=["title", "text"],
            # Options for specifying what to bring from the vector(s)
            # of the looked up point, True by default
            with_vectors=False,
        ),
    )

    #  search batch
    # filter_ = models.Filter(
    #     must=[
    #         models.FieldCondition(
    #             key="city",
    #             match=models.MatchValue(
    #                 value="London",
    #             ),
    #         )
    #     ]
    # )
    #
    # client.query_batch_points(
    #     collection_name="books",
    #     requests=[
    #         models.QueryRequest(
    #             query=models.Document(text="time travel", model="sentence-transformers/all-minilm-l6-v2"),
    #             using="description-dense",
    #             with_payload=True,
    #             filter=models.Filter(
    #                 must=[models.FieldCondition(key="title", match=models.MatchText(text="time travel"))]
    #             ),
    #         ),
    #         models.QueryRequest(
    #             query=models.Document(text="time travel", model="sentence-transformers/all-minilm-l6-v2"),
    #             using="description-dense",
    #             with_payload=True,
    #             filter=models.Filter(
    #                 must=[models.FieldCondition(key="title", match=models.MatchTextAny(text_any="time travel"))]
    #             ),
    #         ),
    #         models.QueryRequest(
    #             query=models.Document(text="time travel", model="sentence-transformers/all-minilm-l6-v2"),
    #             using="description-dense",
    #             with_payload=True,
    #         ),
    #     ],
    # )

    #     group by
    client.facet(
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
print(client.get_collection(collection_name=collection_name))
print(client.get_collection_aliases(collection_name=collection_name))
print(client.get_aliases())
print(client.get_collections())
