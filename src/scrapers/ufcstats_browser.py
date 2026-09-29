"""
Shared Playwright browser for crawling ufcstats.com.

ufcstats pages are rendered with JavaScript, so a plain HTTP fetch returns
no data. One Chromium instance is launched per crawl and reused for every
page; a semaphore bounds how many tabs are open at once so the crawl stays
polite and memory usage stays flat over thousands of pages.

Usage:
    async with UFCStatsBrowser() as browser:
        html = await browser.fetch_rendered(url, "table.b-fight-details__table")
"""

import asyncio
import logging

from playwright.async_api import Browser, BrowserContext, Playwright, async_playwright
from playwright.async_api import Error as PlaywrightError

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class UFCStatsBrowser:
    def __init__(
        self,
        headless: bool = True,
        max_concurrency: int = 4,
        timeout_ms: int = 30000,
        retries: int = 3,
    ) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.retries = retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def __aenter__(self) -> "UFCStatsBrowser":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(user_agent=USER_AGENT)
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._context is not None:
            await self._context.close()
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()

    async def fetch_rendered(self, url: str, wait_selector: str) -> str:
        """Return the page HTML once `wait_selector` has rendered.

        Retries with exponential backoff on timeouts and navigation errors.
        """
        if self._context is None:
            raise RuntimeError("UFCStatsBrowser must be used as an async context manager")

        async with self._semaphore:
            for attempt in range(1, self.retries + 1):
                page = await self._context.new_page()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                    await page.wait_for_selector(wait_selector, timeout=self.timeout_ms)
                    return await page.content()
                except PlaywrightError as exc:
                    if attempt == self.retries:
                        raise
                    delay = 2 ** attempt
                    logger.warning("Fetch failed (%s/%s) for %s: %s; retrying in %ss",
                                   attempt, self.retries, url, exc, delay)
                    await asyncio.sleep(delay)
                finally:
                    await page.close()

        raise AssertionError("unreachable")
