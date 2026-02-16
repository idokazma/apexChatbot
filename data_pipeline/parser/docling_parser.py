"""Parse PDF and HTML files using Docling into structured markdown."""

import json
from pathlib import Path

from docling.document_converter import DocumentConverter
from loguru import logger


class DoclingParser:
    """Parses PDF and HTML files into structured markdown using Docling."""

    def __init__(self):
        self.converter = DocumentConverter()

    def parse_file(self, file_path: Path, source_url: str = "") -> dict | None:
        """Parse a single file (PDF or HTML) into structured content.

        Returns:
            Dict with 'markdown', 'tables', 'metadata' keys, or None on failure.
        """
        try:
            result = self.converter.convert(str(file_path))
            doc = result.document

            # Export full document as markdown
            markdown = doc.export_to_markdown()

            # Extract tables separately for special handling
            tables = []
            for i, table in enumerate(doc.tables):
                try:
                    df = table.export_to_dataframe()
                    tables.append({
                        "index": i,
                        "markdown": df.to_markdown(index=False),
                        "rows": len(df),
                        "columns": list(df.columns),
                    })
                except Exception as e:
                    logger.warning(f"  Failed to export table {i} from {file_path}: {e}")

            return {
                "markdown": markdown,
                "tables": tables,
                "source_file": str(file_path),
                "source_url": source_url,
                "file_type": file_path.suffix.lower().lstrip("."),
            }

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None

    def parse_domain(
        self,
        domain_name: str,
        raw_dir: Path,
        output_dir: Path,
        scrape_results: list[dict],
    ) -> list[dict]:
        """Parse all files for a domain.

        Args:
            domain_name: Insurance domain name.
            raw_dir: Directory containing raw scraped files.
            output_dir: Directory to save parsed output.
            scrape_results: List of scrape result dicts with 'file_path' and 'url'.
        """
        domain_output = output_dir / domain_name
        domain_output.mkdir(parents=True, exist_ok=True)

        parsed_docs = []

        for entry in scrape_results:
            file_path = Path(entry["file_path"])
            if not file_path.exists():
                logger.warning(f"  File not found: {file_path}")
                continue

            result = self.parse_file(file_path, source_url=entry.get("url", ""))
            if result:
                result["title"] = entry.get("title", "")
                result["domain"] = domain_name

                # Save parsed result
                output_path = domain_output / f"{file_path.stem}.json"
                output_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                parsed_docs.append(result)
                logger.debug(f"  Parsed: {file_path.name}")

        logger.info(f"Parsed {len(parsed_docs)} documents for {domain_name}")
        return parsed_docs
