"""Sample random documents from ChromaDB for question generation."""

import random

from loguru import logger

from config.domains import DOMAIN_NAMES
from data_pipeline.store.vector_store import VectorStoreClient


def sample_documents(
    store: VectorStoreClient,
    n_samples: int = 100,
    domains: list[str] | None = None,
    min_content_length: int = 100,
) -> list[dict]:
    """Sample random document chunks from the vector store.

    Fetches chunks across all (or specified) domains, filters by content length,
    and returns a random sample.

    Args:
        store: Connected VectorStoreClient.
        n_samples: Number of document chunks to sample.
        domains: Optional list of domains to sample from. Defaults to all.
        min_content_length: Minimum character length for a chunk to be usable.

    Returns:
        List of dicts, each with keys: content, domain, source_url,
        source_doc_title, section_path, page_number, source_file_path, chunk_id.
    """
    domains = domains or DOMAIN_NAMES
    all_docs: list[dict] = []

    for domain in domains:
        try:
            results = store.collection.get(
                where={"domain": domain},
                include=["documents", "metadatas"],
                limit=5000,
            )
            if not results or not results["ids"]:
                logger.warning(f"No documents found for domain: {domain}")
                continue

            for i, chunk_id in enumerate(results["ids"]):
                content = results["documents"][i] if results["documents"] else ""
                meta = results["metadatas"][i] if results["metadatas"] else {}

                if len(content) < min_content_length:
                    continue

                all_docs.append({
                    "chunk_id": chunk_id,
                    "content": content,
                    "domain": meta.get("domain", domain),
                    "source_url": meta.get("source_url", ""),
                    "source_doc_title": meta.get("source_doc_title", ""),
                    "section_path": meta.get("section_path", ""),
                    "page_number": meta.get("page_number", 0),
                    "source_file_path": meta.get("source_file_path", ""),
                    "summary": meta.get("summary", ""),
                    "keywords": meta.get("keywords", ""),
                })

            logger.info(f"Domain '{domain}': {len(results['ids'])} chunks found")

        except Exception as e:
            logger.error(f"Failed to fetch documents for domain {domain}: {e}")

    logger.info(f"Total usable documents: {len(all_docs)}")

    if len(all_docs) <= n_samples:
        random.shuffle(all_docs)
        return all_docs

    return random.sample(all_docs, n_samples)


def sample_document_groups(
    store: VectorStoreClient,
    n_groups: int = 100,
    docs_per_group: int = 2,
    domains: list[str] | None = None,
) -> list[list[dict]]:
    """Sample groups of related documents for multi-document question generation.

    Each group contains docs_per_group chunks, preferring chunks from the same
    domain to enable cross-document questions.

    Args:
        store: Connected VectorStoreClient.
        n_groups: Number of document groups to create.
        docs_per_group: Number of documents per group.
        domains: Optional domain filter.

    Returns:
        List of document groups (each group is a list of doc dicts).
    """
    total_needed = n_groups * docs_per_group * 2  # oversample
    all_docs = sample_documents(store, n_samples=total_needed, domains=domains)

    # Group by domain for intra-domain questions
    by_domain: dict[str, list[dict]] = {}
    for doc in all_docs:
        by_domain.setdefault(doc["domain"], []).append(doc)

    groups: list[list[dict]] = []
    domain_list = list(by_domain.keys())

    for _ in range(n_groups):
        domain = random.choice(domain_list)
        pool = by_domain.get(domain, [])

        if len(pool) >= docs_per_group:
            group = random.sample(pool, docs_per_group)
        else:
            # Fall back to cross-domain
            group = random.sample(all_docs, min(docs_per_group, len(all_docs)))

        groups.append(group)

    logger.info(f"Created {len(groups)} document groups for question generation")
    return groups
