import os
import asyncio
from typing import Dict
import json

import httpx

from . import utils


async def get_metadata(
    text_url: str, audio_url: str, semaphore: asyncio.Semaphore, save_path: str = None
) -> Dict[str, str] | None:
    """
    Asynchronously get audio metadata from an book URL.

    Args:
        text_url (str): The URL of the book page.
        audio_url (str): The URL of the audiobook page.
        save (bool): Whether to save the metadata as a JSON file.

    Returns:
        Dict[str, str]: The book metadata, containing title, url, duration, author, and narrator's name.
    """
    async with semaphore:
        try:
            text_parser = await utils.get_web_content(text_url)
            audio_parser = await utils.get_web_content(audio_url)
        except httpx.HTTPStatusError:
            return

        author = text_parser.css_first(
            "div.product-price span.text-brand"
        )  # The text source is more reliable than audio one
        duration = audio_parser.css_first(".featu")
        narrator = audio_parser.css_first("i.fa-microphone + a")

        metadata = {
            "audio_url": audio_url,
            "text_url": text_url,
            "title": audio_url.split("/")[-1],
            "author": author.text(strip=True) if author else "",
            "duration": duration.text(strip=True) if duration else "",
            "narrator": narrator.text(strip=True) if narrator else "",
        }

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_path += f"{text_url.split('/')[-1]}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
        else:
            print("Don't save any book's metadata")

        return metadata
