import asyncio
from typing import List, Tuple

import httpx
from playwright.async_api import async_playwright
from selectolax.parser import HTMLParser

from tts_data_pipeline import constants


async def get_text_download_url(name: str) -> str:
    return f"{constants.TEXT_DOWNLOAD_URL}{name}.pdf"


async def get_web_content(url: str) -> HTMLParser:
    """
    Asynchronously fetch HTML content from a given URL.

    Args:
        url (str): The audio URL.

    Returns:
        HTMLParser: Parsed HTML content.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
        return HTMLParser(response.text)


async def get_num_page(url: str) -> int:
    """
    Get the number of pages from a given page.

    Args:
        url (str): The URL of the page

    Returns:
        int: The number of pages in each category
    """
    parser = await get_web_content(url)
    string = parser.css_first(
        "div.pagination span"
    ).text()  # The expect output is "Trang 1 trong X"
    num_page = int(string.split(" ")[-1])  # Get X
    return num_page


async def get_all_audiobook_url() -> Tuple[List[str]]:
    """
    Asynchronously fetch all audiobook URLs from different categories.

    Returns:
        List[str]: A list of all audiobook URLs
    """
    categories = [
        "kinh-te-khoi-nghiep",
        "tam-linh-ton-giao",
        "truyen-tieu-thuyet",
        "tu-duy-ky-nang",
        "tu-lieu-lich-su",
    ]

    category_urls = [
        f"{constants.AUDIO_CATEGORY_URL}{category}" for category in categories
    ]

    # Get the number of page for each category
    num_pages = await asyncio.gather(*(get_num_page(url) for url in category_urls))

    # Get the web content from each category in each page
    page_urls = []
    for url, num_page in zip(category_urls, num_pages):
        page_urls.append(url)
        page_urls.extend([f"{url}/page/{page}" for page in range(2, num_page + 1)])
    parsers = await asyncio.gather(*(get_web_content(url) for url in page_urls))

    # Extract all audiobook URLs from each page
    book_urls = [
        node.attributes.get("href")
        for parser in parsers
        for node in parser.css("div.poster a")
    ]

    return book_urls


async def fetch_download_audio_url(book_url: str) -> List[str]:
    """Fetch all download URLs for a given book using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(book_url)

        # Lấy tất cả các link có class 'ai-track-btn'
        mp3_links = await page.locator("a.ai-track-btn").evaluate_all(
            "elements => elements.map(el => el.href)"
        )

        await browser.close()
        return mp3_links
