import asyncio
import json
import os
from typing import Dict, List, Tuple

import httpx
import requests
from selectolax.parser import HTMLParser
from tqdm import tqdm

from pipeline import utils

SAVEPATH = "../../data/audio/"


async def get_all_audiobook_url() -> List[str]:
    """
    Asynchronously fetch all audiobook URLs from different categories.

    Returns:
        List[str]: List of audiobook URLs.
    """
    categories = [
        "kinh-te-khoi-nghiep",
        "tam-linh-ton-giao",
        "truyen-tieu-thuyet",
        "tu-duy-ki-nang",
        "tu-lieu-lich-su",
    ]

    tasks = []
    base_url = "https://sachnoiviet.net/danh-muc-sach/"

    for category in categories:
        tasks.append(utils.get_web_content(f"{base_url}{category}"))

    pages = await asyncio.gather(*tasks)

    book_urls = []
    for page in pages:
        book_urls.extend(
            node.attributes.get("href") for node in page.css("div.poster a")
        )

    return book_urls


async def get_book_name(url: str) -> str:
    parser = utils.get_web_content(url)
    return parser.css_first("div.data h1").text()
