"""Recursive crawler to discover all pages and PDFs under Harel insurance domains."""

import asyncio
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

from loguru import logger
from playwright.async_api import async_playwright

from config.domains import DOMAINS, InsuranceDomain
from data_pipeline.scraper.rate_limiter import RateLimiter


class SitemapCrawler:
    """Crawls Harel insurance pages to discover all URLs and PDF links per domain."""

    HAREL_BASE = "https://www.harel-group.co.il"

    def __init__(self, output_dir: Path, max_pages_per_domain: int = 200):
        self.output_dir = output_dir
        self.max_pages_per_domain = max_pages_per_domain
        self.rate_limiter = RateLimiter(delay_seconds=1.5)

    async def crawl_all_domains(self) -> dict[str, dict]:
        """Crawl all insurance domains and return discovered URLs."""
        results = {}
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                locale="he-IL",
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )

            for domain_name, domain in DOMAINS.items():
                logger.info(f"Crawling domain: {domain_name} ({domain.base_url})")
                result = await self._crawl_domain(context, domain)
                results[domain_name] = result

                # Save per-domain manifest
                manifest_path = self.output_dir / domain_name / "manifest.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                logger.info(
                    f"  {domain_name}: {len(result['pages'])} pages, "
                    f"{len(result['pdfs'])} PDFs"
                )

            await browser.close()

        return results

    async def _crawl_domain(
        self, context, domain: InsuranceDomain
    ) -> dict:
        """Crawl a single domain, discovering pages and PDF links."""
        visited: set[str] = set()
        to_visit: list[str] = [domain.base_url]
        pages: list[str] = []
        pdfs: list[dict] = []

        while to_visit and len(visited) < self.max_pages_per_domain:
            url = to_visit.pop(0)
            normalized = self._normalize_url(url)

            if normalized in visited:
                continue
            visited.add(normalized)

            await self.rate_limiter.wait()

            try:
                page = await context.new_page()
                response = await page.goto(url, wait_until="networkidle", timeout=30000)

                if not response or response.status != 200:
                    await page.close()
                    continue

                pages.append(url)

                # Extract all links
                links = await page.eval_on_selector_all(
                    "a[href]",
                    "elements => elements.map(el => el.href)",
                )

                for link in links:
                    resolved = urljoin(url, link)
                    resolved_normalized = self._normalize_url(resolved)

                    # Collect PDFs
                    if resolved.lower().endswith(".pdf"):
                        if not any(p["url"] == resolved for p in pdfs):
                            pdfs.append({"url": resolved, "found_on": url})
                        continue

                    # Only follow links within this insurance domain
                    if (
                        self._is_same_domain_section(resolved, domain)
                        and resolved_normalized not in visited
                    ):
                        to_visit.append(resolved)

                await page.close()

            except Exception as e:
                logger.warning(f"  Error crawling {url}: {e}")
                try:
                    await page.close()
                except Exception:
                    pass

        return {
            "domain": domain.name,
            "base_url": domain.base_url,
            "pages": pages,
            "pdfs": pdfs,
            "total_visited": len(visited),
        }

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and trailing slashes."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    def _is_same_domain_section(self, url: str, domain: InsuranceDomain) -> bool:
        """Check if URL belongs to the same insurance domain section."""
        parsed = urlparse(url)
        base_parsed = urlparse(domain.base_url)
        return (
            parsed.netloc == base_parsed.netloc
            and parsed.path.startswith(base_parsed.path.rstrip("/"))
            and not url.lower().endswith((".jpg", ".png", ".gif", ".svg", ".css", ".js"))
        )


async def crawl_all(output_dir: Path) -> dict:
    """Entry point to crawl all domains."""
    crawler = SitemapCrawler(output_dir=output_dir)
    return await crawler.crawl_all_domains()
