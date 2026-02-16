"""ChromaDB collection configuration.

This module defines the collection name and metadata settings for the vector store.
The old Milvus schema (FieldSchema, CollectionSchema) is no longer used.
"""

from config.settings import settings

COLLECTION_NAME = settings.chromadb_collection
