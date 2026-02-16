"""Scrape rendered HTML content from Harel ASPX pages."""

import hashlib
import json
from pathlib import Path

from loguru import logger
from playwright.async_api import async_playwright

from data_pipeline.scraper.rate_limiter import RateLimiter


class AspxScraper:
    """Downloads rendered HTML from ASPX pages using Playwright."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.rate_limiter = RateLimiter(delay_seconds=1.5)

    async def scrape_domain(self, domain_name: str, urls: list[str]) -> list[dict]:
        """Scrape all pages for a domain, saving rendered HTML."""
        domain_dir = self.output_dir / domain_name / "html"
        domain_dir.mkdir(parents=True, exist_ok=True)

        results = []

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

            for url in urls:
                await self.rate_limiter.wait()
                result = await self._scrape_page(context, url, domain_dir)
                if result:
                    results.append(result)
                    logger.debug(f"  Scraped: {url}")
                else:
                    logger.warning(f"  Failed: {url}")

            await browser.close()

        # Save scrape results
        meta_path = self.output_dir / domain_name / "scrape_results.json"
        meta_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        logger.info(f"Scraped {len(results)}/{len(urls)} pages for {domain_name}")

        return results

    async def _scrape_page(self, context, url: str, output_dir: Path) -> dict | None:
        """Scrape a single page and save its HTML."""
        try:
            page = await context.new_page()
            response = await page.goto(url, wait_until="networkidle", timeout=30000)

            if not response or response.status != 200:
                await page.close()
                return None

            # Wait for main content to load
            await page.wait_for_timeout(2000)

            # Get the page title
            title = await page.title()

            # Get the rendered HTML of the main content area
            # Try common content selectors for Harel's site
            content_html = None
            for selector in [
                "main",
                ".main-content",
                "#main-content",
                ".content-area",
                "article",
                ".page-content",
            ]:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        content_html = await element.inner_html()
                        break
                except Exception:
                    continue

            # Fallback to body
            if not content_html:
                content_html = await page.content()

            # Save HTML file
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
            file_path = output_dir / f"{url_hash}.html"
            file_path.write_text(content_html, encoding="utf-8")

            await page.close()

            return {
                "url": url,
                "title": title,
                "file_path": str(file_path),
                "url_hash": url_hash,
            }

        except Exception as e:
            logger.warning(f"  Error scraping {url}: {e}")
            try:
                await page.close()
            except Exception:
                pass
            return None
