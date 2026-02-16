"""Download PDFs discovered during crawling."""

import hashlib
from pathlib import Path

import httpx
from loguru import logger

from data_pipeline.scraper.rate_limiter import RateLimiter


class PdfDownloader:
    """Downloads PDF files from discovered URLs."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.rate_limiter = RateLimiter(delay_seconds=1.0)

    async def download_domain_pdfs(
        self, domain_name: str, pdf_entries: list[dict]
    ) -> list[dict]:
        """Download all PDFs for a domain.

        Args:
            domain_name: Insurance domain name.
            pdf_entries: List of dicts with 'url' and 'found_on' keys.
        """
        pdf_dir = self.output_dir / domain_name / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        results = []
        async with httpx.AsyncClient(
            timeout=60.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36"
                )
            },
        ) as client:
            for entry in pdf_entries:
                url = entry["url"]
                await self.rate_limiter.wait()

                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                        # Use hash-only filename to avoid macOS 255-char limit
                        # with Hebrew URL-encoded filenames
                        file_path = pdf_dir / f"{url_hash}.pdf"
                        file_path.write_bytes(response.content)

                        results.append({
                            "url": url,
                            "found_on": entry.get("found_on", ""),
                            "file_path": str(file_path),
                            "size_bytes": len(response.content),
                        })
                        logger.debug(f"  Downloaded PDF: {url_hash}.pdf ({url[-60:]})")
                    else:
                        logger.warning(f"  PDF download failed ({response.status_code}): {url}")

                except Exception as e:
                    logger.warning(f"  Error downloading PDF {url}: {e}")

        logger.info(
            f"Downloaded {len(results)}/{len(pdf_entries)} PDFs for {domain_name}"
        )
        return results
