import asyncio
import os
from typing import List

import aiofiles
import httpx
from rich import print
from tqdm.asyncio import tqdm

from tts_data_pipeline import constants, utils


async def fetch_download_audio_url(book_url: str) -> List[str]:
    """Fetch all download URLs for a given book."""
    parser = await utils.get_web_content(book_url)
    return [node.attributes.get("href") for node in parser.css("a.ai-track-btn")]


async def fetch_download_text_url(book_url: str) -> str:
    """Fetch the download URL for the text of a given book."""
    parser = await utils.get_web_content(book_url)
    return parser.css_first("a[id=download-pdf]").attributes.get("href")


async def download_audio(url: str, file_path: str):
    """
    Download audio file asynchronously with progress tracking.

    Args:
        url (str): The download URL of the audio file.
        save_path (str): The file path to save the downloaded audio file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", 0))

            async with (
                aiofiles.open(file_path, "wb") as file,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {os.path.basename(file_path)}",
                ) as progress,
            ):
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    await file.write(chunk)
                    progress.update(len(chunk))


async def download_text(url: str, file_path: str):
    """
    Download a PDF file from the given URL.

    Args:
        url (str): The download URL of the PDF.
        save_path (str): The file path to save the downloaded PDF.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))

        async with (
            aiofiles.open(file_path, "wb") as file,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {os.path.basename(file_path)}",
            ) as progress,
        ):
            for chunk in response.iter_bytes(chunk_size=1024):
                file.write(chunk)
                progress.update(len(chunk))


async def download_full_book(
    audio_url: str, text_url: str, audio_savepath: str, text_savepath: str
):
    """Fetch download URLs and download all audio parts of a book."""
    book_name = audio_url.split("/")[-1]
    audio_download_urls = await fetch_download_audio_url(audio_url)
    text_download_url = await fetch_download_text_url(text_url)

    tasks = []
    text_path = os.path.join(text_savepath, f"{book_name}.pdf")
    tasks.append(download_text(text_download_url, text_path))

    for idx, url in enumerate(audio_download_urls):
        audio_path = os.path.join(audio_savepath, f"{book_name}_{idx}.mp3")
        tasks.append(download_audio(url, audio_path))

    await asyncio.gather(*tasks)


async def main():
    """
    Main function to get all audiobook URLs and download them.
    """
    os.makedirs(constants.AUDIO_SAVEPATH, exist_ok=True)

    # Get all audiobook URLs
    print("Getting all audiobook URLs and names")
    all_audio_urls = await utils.audio.get_all_audiobook_url()
    print(f"Found {len(all_audio_urls)} audiobooks")

    # Save all audiobook's URLs
    print(f"Saving all audiobook URLs to {constants.ALL_AUDIOBOOK_URLS_SAVEPATH} file")
    async with aiofiles.open(constants.ALL_AUDIOBOOK_URLS_SAVEPATH, "w") as f:
        for url in all_audio_urls:
            await f.write(url + "\n")

    # Filter books that exist in text source
    audio_urls = [
        url
        for url in tqdm(all_audio_urls, desc="Checking existence from text source")
        if await utils.text.check_exists(url.split("/")[-1])
    ]
    print(
        f"""After checking the existence of the book in the text source,
        found {len(audio_urls)} audiobooks to download"""
    )

    # Get metadata for each book
    print(
        f"Getting metadata for each book and save it to JSON file in {constants.METADATA_SAVEPATH}"
    )
    text_urls = [constants.TEXT_BASE_URL + url.split("/")[-1] for url in audio_urls]
    await asyncio.gather(
        *[
            utils.get_metadata(text_url, audio_url, save=True)
            for audio_url, text_url in tqdm(
                zip(audio_urls, text_urls), desc="Getting metadata"
            )
        ]
    )

    # Download books concurrently
    print("Downloading books concurrently")
    await asyncio.gather(
        *(
            download_full_book(
                audio_url, text_url, constants.AUDIO_SAVEPATH, constants.TEXT_SAVEPATH
            )
            for audio_url, text_url in zip(audio_urls, text_urls)
        )
    )
    print("Download complete!")
