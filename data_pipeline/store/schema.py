"""Milvus collection schema definition."""

from pymilvus import CollectionSchema, DataType, FieldSchema

from config.settings import settings

COLLECTION_NAME = settings.milvus_collection


def get_collection_schema() -> CollectionSchema:
    """Define the Milvus collection schema for insurance document chunks."""
    fields = [
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        ),
        FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=8192,
        ),
        FieldSchema(
            name="content_with_context",
            dtype=DataType.VARCHAR,
            max_length=10240,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=settings.embedding_dim,
        ),
        FieldSchema(
            name="domain",
            dtype=DataType.VARCHAR,
            max_length=32,
        ),
        FieldSchema(
            name="source_url",
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
        FieldSchema(
            name="source_doc_title",
            dtype=DataType.VARCHAR,
            max_length=256,
        ),
        FieldSchema(
            name="section_path",
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
        FieldSchema(
            name="language",
            dtype=DataType.VARCHAR,
            max_length=8,
        ),
        FieldSchema(
            name="doc_type",
            dtype=DataType.VARCHAR,
            max_length=32,
        ),
        FieldSchema(
            name="page_number",
            dtype=DataType.INT64,
        ),
        FieldSchema(
            name="chunk_index",
            dtype=DataType.INT64,
        ),
    ]

    return CollectionSchema(
        fields=fields,
        description="Harel Insurance document chunks with embeddings",
    )
