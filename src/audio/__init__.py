import json
import os
from typing import Dict, List, Tuple

import httpx
import pandas as pd
import requests
import selectolax as slx
from selectolax.parser import HTMLParser
from tqdm import tqdm
from src.log import logger


def get_all_book_url() -> List[Tuple[str]]:
    """
    Get all audiobook names and its author

    Returns:
        List[Tuple[str]]: List of tuples containing name and author
    """
    categories = [
        "kinh-te-khoi-nghiep",
        "tam-linh-ton-giao",
        "truyen-tieu-thuyet",
        "tu-duy-ki-nang",
        "tu-lieu-lich-su",
    ]

    book_url = []
    for category in categories:
        catogory_url = f"https://sachnoiviet.net/danh-muc-sach/{category}"
        html = get_content(catogory_url)
        book_url.extend(
            [node.attributes.get("href") for node in html.css("div.poster a")]
        )
    return book_url


def get_content(url: str) -> HTMLParser:
    """
    Get audio content from url

    Args:
        url (str): The audio url

    Returns:
        HTMLParser: The audio content
    """
    html = requests.get(url).text()
    return HTMLParser(html)


def get_metadata(book_url: str, save: bool = False) -> Dict[str, str]:
    """
    Get audio metadata from audio url

    Args:
        book_url (str): The URL of the audiobook page
        save_path (str | None): The path to save the metadata as a JSON file

    Returns:
        pd.DataFrame: The audiobook metadata, containing title, url, duration, author and narrator's name.
    """
    parser = get_content(book_url)
    metadata = {
        "url": book_url,
        "title": parser.css_first("div.data h1").text(),
        "duration": parser.css_first(".featu").text(strip=True),
        "author": parser.css_first("i.fa-user").next.text(),
        "narrator": parser.css_first("i.fa-microphone").next.text(),
    }

    if save:
        save_path = "../../data/audio/"
        os.makedirs(save_path, exist_ok=True)
        save_path += f"{metadata['url'].split('/')[-1]}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            logger.info(f"Saving metadata to {save_path}")
            json.dump(metadata, f, indent=4, ensure_ascii=False)
    else:
        logger.info("Don't save any metadata")

    return metadata


def download_audio(book_url: str) -> List[str]:
    """
    Get audio download url

    Args:
        parser (HTMLParser): The content parser

    Returns:
        List[str]: List of download url, it contains some audio parts
    """

    # Get audio download url
    parser = get_content(book_url)
    download_urls = [
        node.attributes.get("href") for node in parser.css("a.ai-track-btn")
    ]

    # Setup saving path and filename
    save_path = "../../data/audio/"
    os.makedirs(save_path, exist_ok=True)
    filename = book_url.split("/")[-1]

    # Download audio
    for idx, url in enumerate(download_urls):
        with httpx.stream("GET", url) as response:
            response.raise_for_status()
            file_path = f"{save_path}{filename}_{idx}.mp3"
            total_size = int(response.headers.get("content-length", 0))

            with (
                open(file_path, "wb") as file,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {filename}_{idx}",
                ) as progress,
            ):
                for chunk in response.iter_bytes(chunk_size=1024):
                    file.write(chunk)
                    progress.update(len(chunk))
        logger.info(f"Downloaded: {filename}")
