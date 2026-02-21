"""Level 0 builder: generate the library catalog from domain summaries.

Produces a single catalog.json that serves as the entry point for
agentic navigation — the LLM reads this first to pick domains.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from data_pipeline.hierarchy.hierarchy_models import (
    DomainOverview,
    DomainShelf,
    LibraryCatalog,
)
from llm.claude_client import ClaudeClient
from llm.gemini_client import GeminiClient

_CATALOG_PROMPT = """You are building the top-level catalog for an insurance company knowledge base used by a search agent.

The knowledge base has {domain_count} insurance domains:

{domains_text}

For each domain, generate a concise 2-3 sentence summary and a list of 3-5 example question types that belong in that domain.

Generate a JSON object with:
{{
  "domains": [
    {{
      "domain": "<domain name>",
      "summary": "<2-3 sentence summary>",
      "handles_questions_like": ["<example question type 1>", "<example question type 2>", ...]
    }},
    ...
  ]
}}

Include both Hebrew and English in the example questions where relevant.
Respond with ONLY the JSON object, no markdown fences."""


def build_catalog(
    domain_summaries: list[DomainShelf],
    output_dir: Path,
    claude_client: ClaudeClient | None = None,
    gemini_client: GeminiClient | None = None,
) -> LibraryCatalog:
    """Build Level 0 library catalog from domain summaries.

    Args:
        domain_summaries: All domain summaries from Level 1.
        output_dir: Where to write catalog.json.
        claude_client: Claude client (created if None and no Gemini).
        gemini_client: Gemini client (used as fallback if Claude unavailable).

    Returns:
        LibraryCatalog object.
    """
    client = claude_client or gemini_client
    if client is None:
        try:
            client = ClaudeClient()
        except Exception:
            client = GeminiClient()

    logger.info(f"Building library catalog from {len(domain_summaries)} domains...")

    # Build domains text for prompt
    domain_parts = []
    for ds in domain_summaries:
        parts = [f"Domain: {ds.domain} ({ds.domain_he})"]
        if ds.overview:
            parts.append(f"  Overview: {ds.overview}")
        parts.append(f"  Documents: {ds.total_documents}, Chunks: {ds.total_chunks}")
        group_names = [g.group_name for g in ds.document_groups]
        if group_names:
            parts.append(f"  Document groups: {', '.join(group_names)}")
        domain_parts.append("\n".join(parts))
    domains_text = "\n\n".join(domain_parts)

    prompt = _CATALOG_PROMPT.format(
        domain_count=len(domain_summaries),
        domains_text=domains_text,
    )

    try:
        response = client.generate(prompt, temperature=0.0, max_tokens=4096)
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        result = json.loads(response)
    except Exception as e:
        logger.warning(f"Catalog generation failed: {e}")
        result = {"domains": []}

    # Merge LLM output with known metadata
    domain_map = {ds.domain: ds for ds in domain_summaries}
    domain_overviews = []
    for entry in result.get("domains", []):
        name = entry.get("domain", "")
        ds = domain_map.get(name)
        domain_overviews.append(
            DomainOverview(
                domain=name,
                domain_he=ds.domain_he if ds else "",
                summary=entry.get("summary", ds.overview if ds else ""),
                handles_questions_like=entry.get("handles_questions_like", []),
                document_count=ds.total_documents if ds else 0,
            )
        )

    # Add any domains the LLM missed
    seen = {d.domain for d in domain_overviews}
    for ds in domain_summaries:
        if ds.domain not in seen:
            domain_overviews.append(
                DomainOverview(
                    domain=ds.domain,
                    domain_he=ds.domain_he,
                    summary=ds.overview,
                    handles_questions_like=[g.group_name for g in ds.document_groups[:5]],
                    document_count=ds.total_documents,
                )
            )

    total_docs = sum(ds.total_documents for ds in domain_summaries)

    catalog = LibraryCatalog(
        domains=domain_overviews,
        total_documents=total_docs,
        total_domains=len(domain_summaries),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    # Write to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "catalog.json").write_text(
        catalog.model_dump_json(indent=2), encoding="utf-8"
    )

    logger.info(f"Library catalog built: {len(domain_overviews)} domains, {total_docs} total documents")
    return catalog
