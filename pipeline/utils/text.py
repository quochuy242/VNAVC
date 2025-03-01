import asyncio
import json
import os
from typing import Dict

import aiofiles
import httpx
from tqdm.asyncio import tqdm

from pipeline import utils

SAVEPATH = "../../data/text/"
TEXT_BASE_URL = "https://tiemsach.org/"


async def check_exists(name: str) -> bool:
    """Check if a book exists in the text source."""
    url = TEXT_BASE_URL + name.replace(" ", "-").lower()

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.status_code == 200
