"""Data Explorer API — browse hierarchy, documents, and chunks from JSON files."""

import json
import re
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, Query
from loguru import logger

router = APIRouter(prefix="/explorer", tags=["explorer"])

# ── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHUNKS_FILE = DATA_DIR / "chunks" / "all_chunks.json"
PARSED_DIR = DATA_DIR / "parsed"
HIERARCHY_DIR = DATA_DIR / "hierarchy"

# ── Domain Hebrew names ─────────────────────────────────────────────────────
DOMAIN_HE = {
    "car": "רכב",
    "life": "חיים",
    "travel": 'נסיעות לחו"ל',
    "health": "בריאות",
    "dental": "שיניים",
    "mortgage": "משכנתא",
    "business": "עסקים",
    "apartment": "דירה",
}
ALL_DOMAINS = list(DOMAIN_HE.keys())

# ── Lazy-loaded cache ───────────────────────────────────────────────────────
_cache: dict = {}


def _load_chunks() -> list[dict]:
    """Load all_chunks.json once and cache it."""
    if "chunks" not in _cache:
        logger.info("Explorer: loading all_chunks.json ...")
        with open(CHUNKS_FILE) as f:
            _cache["chunks"] = json.load(f)
        _build_indexes()
        logger.info(f"Explorer: loaded {len(_cache['chunks'])} chunks")
    return _cache["chunks"]


def _build_indexes():
    """Build in-memory indexes for fast filtering."""
    chunks = _cache["chunks"]
    by_domain: dict[str, list[dict]] = {}
    by_doc: dict[str, list[dict]] = {}
    by_id: dict[str, dict] = {}

    for c in chunks:
        meta = c.get("metadata", {})
        domain = meta.get("domain", "unknown")
        doc_id = meta.get("source_doc_id", "unknown")
        chunk_id = meta.get("chunk_id", "")

        by_domain.setdefault(domain, []).append(c)
        by_doc.setdefault(f"{domain}/{doc_id}", []).append(c)
        if chunk_id:
            by_id[chunk_id] = c

    # Build doc_id → resolved display title mapping
    doc_titles: dict[str, str] = {}
    for c in chunks:
        meta = c.get("metadata", {})
        doc_id = meta.get("source_doc_id", "unknown")
        if doc_id not in doc_titles:
            doc_titles[doc_id] = _resolve_title(meta)

    _cache["by_domain"] = by_domain
    _cache["by_doc"] = by_doc
    _cache["by_id"] = by_id
    _cache["doc_titles"] = doc_titles


_HASH_RE = re.compile(r"^[0-9a-f]{8,}$")


def _resolve_title(meta: dict) -> str:
    """Derive a human-readable title from chunk metadata.

    Priority: source_doc_title (if not a hash) → source_file_path filename
    (cleaned) → URL-decoded last segment of source_url → doc_id.
    """
    title = meta.get("source_doc_title", "")
    if title and not _HASH_RE.match(title):
        return title

    # Try source_file_path: "car/files/פוליסת-ביטוח-רכב.pdf" → "פוליסת ביטוח רכב"
    sfp = meta.get("source_file_path", "")
    if sfp:
        name = Path(sfp).stem  # strip extension
        name = name.replace("-", " ").replace("_", " ").strip()
        if name and not _HASH_RE.match(name):
            return name

    # Try last meaningful segment of source_url (URL-decoded)
    url = meta.get("source_url", "")
    if url:
        last_seg = unquote(url.rstrip("/").rsplit("/", 1)[-1])
        last_seg = Path(last_seg).stem.replace("-", " ").replace("_", " ").strip()
        if last_seg and not _HASH_RE.match(last_seg):
            return last_seg

    return meta.get("source_doc_id", title or "untitled")


def _is_enriched(chunk: dict) -> str:
    """Return enrichment status: 'full', 'partial', or 'none'."""
    meta = chunk.get("metadata", {})
    has_summary = bool(meta.get("summary"))
    has_keywords = bool(meta.get("keywords"))
    has_facts = bool(meta.get("key_facts"))
    count = sum([has_summary, has_keywords, has_facts])
    if count == 3:
        return "full"
    if count > 0:
        return "partial"
    return "none"


def _get_doc_title(doc_id: str) -> str:
    """Look up the resolved display title for a doc_id."""
    return _cache.get("doc_titles", {}).get(doc_id, doc_id)


def _list_parsed_docs(domain: str) -> list[dict]:
    """List parsed doc files for a domain."""
    domain_dir = PARSED_DIR / domain
    if not domain_dir.exists():
        return []
    docs = []
    for f in sorted(domain_dir.iterdir()):
        if f.suffix == ".json":
            docs.append({"doc_id": f.stem, "domain": domain, "file": str(f)})
    return docs


def _read_parsed_doc(domain: str, doc_id: str) -> dict | None:
    """Read a single parsed document."""
    path = PARSED_DIR / domain / f"{doc_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/stats")
async def get_stats():
    """Overall statistics."""
    chunks = _load_chunks()
    by_domain = _cache["by_domain"]

    total_enriched = sum(1 for c in chunks if _is_enriched(c) != "none")
    total_docs = len(set(c["metadata"]["source_doc_id"] for c in chunks))
    total_parsed = sum(
        len(list((PARSED_DIR / d).glob("*.json")))
        for d in ALL_DOMAINS
        if (PARSED_DIR / d).exists()
    )

    return {
        "total_chunks": len(chunks),
        "total_docs_in_chunks": total_docs,
        "total_parsed_docs": total_parsed,
        "total_enriched": total_enriched,
        "enrichment_pct": round(total_enriched / len(chunks) * 100, 1) if chunks else 0,
        "domains": len(by_domain),
        "domain_counts": {
            d: len(by_domain.get(d, [])) for d in ALL_DOMAINS
        },
    }


@router.get("/domains")
async def list_domains():
    """List all domains with counts."""
    chunks = _load_chunks()
    by_domain = _cache["by_domain"]

    result = []
    for domain in ALL_DOMAINS:
        domain_chunks = by_domain.get(domain, [])
        doc_ids = set(c["metadata"]["source_doc_id"] for c in domain_chunks)
        enriched = sum(1 for c in domain_chunks if _is_enriched(c) != "none")
        parsed_count = len(list((PARSED_DIR / domain).glob("*.json"))) if (PARSED_DIR / domain).exists() else 0

        result.append({
            "domain": domain,
            "domain_he": DOMAIN_HE.get(domain, domain),
            "chunk_count": len(domain_chunks),
            "doc_count": len(doc_ids),
            "parsed_doc_count": parsed_count,
            "enriched_count": enriched,
            "enrichment_pct": round(enriched / len(domain_chunks) * 100, 1) if domain_chunks else 0,
        })

    return result


@router.get("/domains/{domain}/documents")
async def list_domain_documents(domain: str):
    """List documents in a domain with chunk counts."""
    _load_chunks()
    by_domain = _cache["by_domain"]
    domain_chunks = by_domain.get(domain, [])

    # Group chunks by doc_id
    docs: dict[str, dict] = {}
    for c in domain_chunks:
        meta = c["metadata"]
        doc_id = meta["source_doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "domain": domain,
                "title": _get_doc_title(doc_id),
                "doc_type": meta.get("doc_type", "unknown"),
                "chunk_count": 0,
                "enriched_count": 0,
                "has_parsed": (PARSED_DIR / domain / f"{doc_id}.json").exists(),
            }
        docs[doc_id]["chunk_count"] += 1
        if _is_enriched(c) != "none":
            docs[doc_id]["enriched_count"] += 1

    return sorted(docs.values(), key=lambda d: d["title"])


@router.get("/documents/{domain}/{doc_id}")
async def get_document(domain: str, doc_id: str):
    """Full parsed document content + metadata."""
    doc = _read_parsed_doc(domain, doc_id)
    if doc is None:
        return {"error": "Document not found", "domain": domain, "doc_id": doc_id}

    # Get chunk count for this doc
    _load_chunks()
    doc_chunks = _cache["by_doc"].get(f"{domain}/{doc_id}", [])

    return {
        "doc_id": doc_id,
        "domain": domain,
        "title": _get_doc_title(doc_id),
        "source_url": doc.get("source_url", ""),
        "file_type": doc.get("file_type", ""),
        "source_file": doc.get("source_file", ""),
        "markdown": doc.get("markdown", ""),
        "page_count": len(doc.get("page_map", [])) if doc.get("page_map") else None,
        "tables": len(doc.get("tables", [])),
        "chunk_count": len(doc_chunks),
    }


@router.get("/documents/{domain}/{doc_id}/chunks")
async def get_document_chunks(domain: str, doc_id: str):
    """All chunks for a specific document."""
    _load_chunks()
    doc_chunks = _cache["by_doc"].get(f"{domain}/{doc_id}", [])

    result = []
    for c in sorted(doc_chunks, key=lambda x: x["metadata"].get("chunk_index", 0)):
        meta = c["metadata"]
        result.append({
            "chunk_id": meta.get("chunk_id", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "content": c.get("content", ""),
            "content_with_context": c.get("content_with_context", ""),
            "token_count": c.get("token_count", 0),
            "section_path": meta.get("section_path", ""),
            "page_number": meta.get("page_number"),
            "summary": meta.get("summary", ""),
            "keywords": meta.get("keywords", []),
            "key_facts": meta.get("key_facts", []),
            "enrichment_status": _is_enriched(c),
        })

    return {"domain": domain, "doc_id": doc_id, "total": len(result), "chunks": result}


@router.get("/chunks")
async def list_chunks(
    domain: str | None = Query(None),
    doc_type: str | None = Query(None),
    enriched_only: bool = Query(False),
    search: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """Paginated chunk listing with filters."""
    chunks = _load_chunks()

    filtered = chunks
    if domain:
        filtered = [c for c in filtered if c["metadata"].get("domain") == domain]
    if doc_type:
        filtered = [c for c in filtered if c["metadata"].get("doc_type") == doc_type]
    if enriched_only:
        filtered = [c for c in filtered if _is_enriched(c) != "none"]
    if search:
        search_lower = search.lower()
        filtered = [
            c for c in filtered
            if search_lower in c.get("content", "").lower()
            or search_lower in c.get("metadata", {}).get("summary", "").lower()
            or any(search_lower in kw.lower() for kw in c.get("metadata", {}).get("keywords", []))
        ]

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_chunks = filtered[start:end]

    result = []
    for c in page_chunks:
        meta = c["metadata"]
        result.append({
            "chunk_id": meta.get("chunk_id", ""),
            "domain": meta.get("domain", ""),
            "source_doc_id": meta.get("source_doc_id", ""),
            "source_doc_title": _get_doc_title(meta.get("source_doc_id", "")),
            "doc_type": meta.get("doc_type", ""),
            "section_path": meta.get("section_path", ""),
            "page_number": meta.get("page_number"),
            "token_count": c.get("token_count", 0),
            "content_preview": c.get("content", "")[:200],
            "summary": meta.get("summary", ""),
            "keywords": meta.get("keywords", []),
            "key_facts": meta.get("key_facts", []),
            "enrichment_status": _is_enriched(c),
        })

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "chunks": result,
    }


@router.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: str):
    """Single chunk full detail."""
    _load_chunks()
    chunk = _cache["by_id"].get(chunk_id)
    if not chunk:
        return {"error": "Chunk not found", "chunk_id": chunk_id}

    meta = chunk["metadata"]
    return {
        "chunk_id": meta.get("chunk_id", ""),
        "domain": meta.get("domain", ""),
        "source_doc_id": meta.get("source_doc_id", ""),
        "source_doc_title": _get_doc_title(meta.get("source_doc_id", "")),
        "doc_type": meta.get("doc_type", ""),
        "section_path": meta.get("section_path", ""),
        "page_number": meta.get("page_number"),
        "chunk_index": meta.get("chunk_index", 0),
        "total_chunks_in_doc": meta.get("total_chunks_in_doc", 0),
        "language": meta.get("language", ""),
        "source_url": meta.get("source_url", ""),
        "source_file_path": meta.get("source_file_path", ""),
        "content": chunk.get("content", ""),
        "content_with_context": chunk.get("content_with_context", ""),
        "token_count": chunk.get("token_count", 0),
        "summary": meta.get("summary", ""),
        "keywords": meta.get("keywords", []),
        "key_facts": meta.get("key_facts", []),
        "enrichment_status": _is_enriched(chunk),
    }


@router.get("/hierarchy/catalog")
async def get_hierarchy_catalog():
    """Catalog overview — generated from chunks data."""
    _load_chunks()
    by_domain = _cache["by_domain"]

    domains = []
    for domain in ALL_DOMAINS:
        domain_chunks = by_domain.get(domain, [])
        doc_ids = set(c["metadata"]["source_doc_id"] for c in domain_chunks)
        enriched = sum(1 for c in domain_chunks if _is_enriched(c) != "none")

        # Check for hierarchy sections
        sections_dir = HIERARCHY_DIR / "sections" / domain
        section_count = 0
        if sections_dir.exists():
            for doc_dir in sections_dir.iterdir():
                if doc_dir.is_dir():
                    section_count += len(list(doc_dir.glob("*.json")))

        domains.append({
            "domain": domain,
            "domain_he": DOMAIN_HE.get(domain, domain),
            "doc_count": len(doc_ids),
            "chunk_count": len(domain_chunks),
            "enriched_count": enriched,
            "enrichment_pct": round(enriched / len(domain_chunks) * 100, 1) if domain_chunks else 0,
            "hierarchy_sections": section_count,
        })

    return {"domains": domains, "total_domains": len(domains)}


@router.get("/hierarchy/domains/{domain}")
async def get_hierarchy_domain(domain: str):
    """Domain shelf — list documents with hierarchy sections."""
    _load_chunks()
    by_domain = _cache["by_domain"]
    domain_chunks = by_domain.get(domain, [])

    # Group by doc
    docs: dict[str, dict] = {}
    for c in domain_chunks:
        meta = c["metadata"]
        doc_id = meta["source_doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "title": _get_doc_title(doc_id),
                "doc_type": meta.get("doc_type", "unknown"),
                "chunk_count": 0,
                "sections": [],
            }
        docs[doc_id]["chunk_count"] += 1

    # Load hierarchy sections if available
    sections_dir = HIERARCHY_DIR / "sections" / domain
    if sections_dir.exists():
        for doc_dir in sections_dir.iterdir():
            if doc_dir.is_dir():
                doc_id = doc_dir.name
                if doc_id in docs:
                    sections = []
                    for sf in sorted(doc_dir.glob("*.json")):
                        try:
                            with open(sf) as f:
                                sec = json.load(f)
                            sections.append({
                                "section_id": sec.get("section_id", sf.stem),
                                "section_path": sec.get("section_path", ""),
                                "summary": sec.get("summary", ""),
                                "topics": sec.get("topics", [])[:5],
                                "chunk_count": sec.get("chunk_count", 0),
                            })
                        except Exception:
                            pass
                    docs[doc_id]["sections"] = sections

    return {
        "domain": domain,
        "domain_he": DOMAIN_HE.get(domain, domain),
        "documents": sorted(docs.values(), key=lambda d: d["title"]),
    }


@router.get("/hierarchy/documents/{domain}/{doc_id}")
async def get_hierarchy_document(domain: str, doc_id: str):
    """Document card — sections from hierarchy data."""
    _load_chunks()
    doc_chunks = _cache["by_doc"].get(f"{domain}/{doc_id}", [])

    sections = []
    sections_dir = HIERARCHY_DIR / "sections" / domain / doc_id
    if sections_dir.exists():
        for sf in sorted(sections_dir.glob("*.json")):
            try:
                with open(sf) as f:
                    sec = json.load(f)
                sections.append(sec)
            except Exception:
                pass

    # Compute doc-level stats
    enriched = sum(1 for c in doc_chunks if _is_enriched(c) != "none")
    all_keywords: set[str] = set()
    for c in doc_chunks:
        all_keywords.update(c.get("metadata", {}).get("keywords", []))

    return {
        "doc_id": doc_id,
        "domain": domain,
        "domain_he": DOMAIN_HE.get(domain, domain),
        "chunk_count": len(doc_chunks),
        "enriched_count": enriched,
        "all_keywords": sorted(all_keywords)[:30],
        "sections": sections,
        "has_parsed": (PARSED_DIR / domain / f"{doc_id}.json").exists(),
    }
