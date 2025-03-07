import json
import os
from typing import Dict

import httpx
from selectolax.parser import HTMLParser
from . import audio, text


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


async def get_metadata(
    text_url: str, audio_url: str, save: bool = False
) -> Dict[str, str]:
    """
    Asynchronously get audio metadata from an book URL.

    Args:
        text_url (str): The URL of the book page.
        audio_url (str): The URL of the audiobook page.
        save (bool): Whether to save the metadata as a JSON file.

    Returns:
        Dict[str, str]: The book metadata, containing title, url, duration, author, and narrator's name.
    """
    # // TODO: Change the css selector for tiemsach.org, not sachnoiviet
    text_parser = await get_web_content(text_url)
    audio_parser = await get_web_content(audio_url)

    metadata = {
        "audio_url": audio_url,
        "text_url": text_url,
        "title": audio_url.split("/")[-1],
        "author": text_parser.css_first("div.book-metadata a strong").text(),
        "duration": audio_parser.css_first(".featu").text(strip=True),
        "narrator": audio_parser.css_first("i.fa-microphone").next.text(),
    }

    if save:
        save_path = "../../data/audio/"
        os.makedirs(save_path, exist_ok=True)
        save_path += f"{metadata['url'].split('/')[-1]}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            print(f"Saving book's metadata to {save_path}")
            json.dump(metadata, f, indent=4, ensure_ascii=False)
    else:
        print("Don't save any book's metadata")

    return metadata
