import asyncio
from typing import List, Tuple

from tts_data_pipeline import utils, constants


async def get_num_page(url: str) -> int:
    """
    Get the number of pages from a given page.

    Args:
        url (str): The URL of the page

    Returns:
        int: The number of pages in each category
    """
    parser = await utils.get_web_content(url)
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
    parsers = await asyncio.gather(*(utils.get_web_content(url) for url in page_urls))

    # Extract all audiobook URLs from each page
    book_urls = [
        node.attributes.get("href")
        for parser in parsers
        for node in parser.css("div.poster a")
    ]

    return book_urls
