"""UFCStatsBrowser error handling, with Playwright's objects faked."""

import asyncio

import pytest
from playwright.async_api import Error as PlaywrightError

from src.scrapers.ufcstats_browser import BrowserClosedError, UFCStatsBrowser


class FakeBrowser:
    def __init__(self, connected: bool) -> None:
        self.connected = connected

    def is_connected(self) -> bool:
        return self.connected


class FakePage:
    def __init__(self, browser: FakeBrowser, dies_on_goto: bool) -> None:
        self.browser = browser
        self.dies_on_goto = dies_on_goto
        self.closed = False

    async def goto(self, url, **kwargs):
        if self.dies_on_goto:
            self.browser.connected = False
        raise PlaywrightError("Target page, context or browser has been closed")

    async def close(self):
        if not self.browser.connected:
            raise PlaywrightError("Target page, context or browser has been closed")
        self.closed = True


class FakeContext:
    def __init__(self, browser: FakeBrowser, dies_on_goto: bool = False) -> None:
        self.browser = browser
        self.dies_on_goto = dies_on_goto
        self.pages: list[FakePage] = []

    async def new_page(self):
        if not self.browser.connected:
            raise PlaywrightError("BrowserContext.new_page: Target page, context or browser has been closed")
        page = FakePage(self.browser, self.dies_on_goto)
        self.pages.append(page)
        return page


def browser_with(context: FakeContext) -> UFCStatsBrowser:
    browser = UFCStatsBrowser(retries=3)
    browser._browser = context.browser
    browser._context = context
    return browser


def test_dead_browser_raises_browser_closed_error():
    browser = browser_with(FakeContext(FakeBrowser(connected=False)))

    with pytest.raises(BrowserClosedError):
        asyncio.run(browser.fetch_rendered("http://ufcstats.com", "body"))


def test_browser_dying_mid_fetch_stops_without_retrying():
    context = FakeContext(FakeBrowser(connected=True), dies_on_goto=True)

    with pytest.raises(BrowserClosedError):
        asyncio.run(browser_with(context).fetch_rendered("http://ufcstats.com", "body"))
    assert len(context.pages) == 1  # no retry, and the dead page is not closed


def test_page_errors_on_a_live_browser_are_retried(monkeypatch):
    async def no_sleep(delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    context = FakeContext(FakeBrowser(connected=True))

    with pytest.raises(PlaywrightError) as raised:
        asyncio.run(browser_with(context).fetch_rendered("http://ufcstats.com", "body"))
    assert not isinstance(raised.value, BrowserClosedError)
    assert len(context.pages) == 3 and all(page.closed for page in context.pages)
