"""Playwright fallback.

Used only when normal fetching cannot retrieve useful content (JS-rendered
sites, Cloudflare interstitials). The crawler calls this lazily so that a run
without Playwright installed still works.
"""

from __future__ import annotations

import asyncio

from leadfinder.crawl.crawler import Page, _clean_text, _extract_links
from leadfinder.crawl.extract import extract_facts
from leadfinder.search.base import classify_source_type

_blocked_markers = (
    "enable javascript", "please enable javascript", "just a moment",
    "checking your browser", "cf-browser-verification", "access denied",
    "are you a robot", "請稍候", "checking your browser before accessing",
)


def playwright_available() -> bool:
    try:
        import playwright.async_api  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


async def render_with_playwright(url: str, crawler) -> Page | None:
    """Render one URL in a headless browser. Returns None if unavailable."""
    if not playwright_available():
        return None
    try:
        return await asyncio.wait_for(_render(url, crawler), timeout=45.0)
    except Exception:
        return None


async def _render(url: str, crawler) -> Page | None:
    from playwright.async_api import async_playwright  # type: ignore

    if crawler._playwright is None:
        crawler._playwright = await async_playwright().start()
    playwright = crawler._playwright
    browser = await playwright.chromium.launch(headless=True)
    try:
        context = await browser.new_context(
            user_agent=crawler.config.crawler.user_agent,
            locale="en-US",
            viewport={"width": 1440, "height": 900},
        )
        page = await context.new_page()
        response = await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=int(crawler.config.crawler.timeout * 1000),
        )
        try:
            await page.wait_for_load_state("networkidle", timeout=6000)
        except Exception:
            pass
        html = await page.content()
        status = response.status if response else None
        # A bot-wall page is long but useless; treat it as a failure so the
        # caller keeps the original HTTP result.
        head = (html or "")[:4000].lower()
        if any(marker in head for marker in _blocked_markers):
            return None
        title, text = _clean_text(html, url)
        if len(text) < 80:
            return None
        result = Page(
            url=url,
            final_url=page.url,
            status_code=status if status else 200,
            title=title,
            text=text,
            html=html,
            source_type=classify_source_type(page.url, title),
            via="playwright",
        )
        result.links = _extract_links(url, page.url, html)
        result.facts = extract_facts(page.url, html, text)
        return result
    finally:
        await browser.close()
