"""Extract and enrich metadata from parsed documents."""

import re
from urllib.parse import urlparse

from langdetect import detect

from config.domains import DOMAIN_NAMES_HE, DOMAINS


def detect_language(text: str) -> str:
    """Detect if text is Hebrew or English."""
    try:
        lang = detect(text[:1000])
        return "he" if lang == "he" else "en"
    except Exception:
        # Default to Hebrew since most Harel content is Hebrew
        return "he"


def detect_doc_type(url: str, title: str) -> str:
    """Classify document type from URL and title patterns."""
    url_lower = url.lower()
    title_lower = title.lower()

    if url_lower.endswith(".pdf"):
        if any(term in title_lower for term in ["פוליסה", "policy", "תנאים כלליים"]):
            return "policy_document"
        return "pdf"

    if any(term in url_lower for term in ["faq", "שאלות"]):
        return "faq"

    return "webpage"


def extract_domain_from_url(url: str) -> str | None:
    """Extract insurance domain from a Harel URL."""
    parsed = urlparse(url)
    path = parsed.path.lower()

    for domain_name, domain in DOMAINS.items():
        domain_path = urlparse(domain.base_url).path.lower()
        if path.startswith(domain_path):
            return domain_name

    return None


def extract_section_path(markdown: str) -> list[str]:
    """Extract heading hierarchy from markdown content."""
    headings = []
    for line in markdown.split("\n"):
        line = line.strip()
        match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append({"level": level, "text": text})
    return headings


def enrich_metadata(parsed_doc: dict) -> dict:
    """Add derived metadata to a parsed document."""
    markdown = parsed_doc.get("markdown", "")
    url = parsed_doc.get("source_url", "")
    title = parsed_doc.get("title", "")

    parsed_doc["language"] = detect_language(markdown)
    parsed_doc["doc_type"] = detect_doc_type(url, title)
    parsed_doc["headings"] = extract_section_path(markdown)

    # Try to extract domain from URL if not set
    if not parsed_doc.get("domain"):
        parsed_doc["domain"] = extract_domain_from_url(url) or "unknown"

    return parsed_doc
