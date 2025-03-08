import asyncio
import os
from typing import List, Tuple

import aiofiles
import httpx
import aiohttp
from rich import print
from tqdm.asyncio import tqdm, tqdm_asyncio

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
        file_path (str): The file path to save the downloaded audio file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", 0))

            async with aiofiles.open(file_path, "wb") as file:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {os.path.basename(file_path)}",
                ) as progress:
                    async for chunk in response.aiter_bytes(
                        chunk_size=8192
                    ):  # Increased chunk size
                        await file.write(chunk)
                        progress.update(len(chunk))


async def download_text(url: str, file_path: str):
    """
    Download a PDF file from the given URL.

    Args:
        url (str): The download URL of the PDF.
        file_path (str): The file path to save the downloaded PDF.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET", url
        ) as response:  # Changed to stream for consistency
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", 0))

            async with aiofiles.open(file_path, "wb") as file:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {os.path.basename(file_path)}",
                ) as progress:
                    async for chunk in response.aiter_bytes(
                        chunk_size=8192
                    ):  # Fixed to use async iteration
                        await file.write(chunk)  # Added await
                        progress.update(len(chunk))


async def download_full_book(
    audio_url: str, text_url: str, audio_savepath: str, text_savepath: str
):
    """Fetch download URLs and download all audio parts of a book."""
    book_name = audio_url.split("/")[-1]

    # Fetch both URLs concurrently
    audio_download_urls, text_download_url = await asyncio.gather(
        fetch_download_audio_url(audio_url), fetch_download_text_url(text_url)
    )

    tasks = []
    text_path = os.path.join(text_savepath, f"{book_name}.pdf")
    tasks.append(download_text(text_download_url, text_path))

    for idx, url in enumerate(audio_download_urls):
        audio_path = os.path.join(audio_savepath, f"{book_name}_{idx}.mp3")
        tasks.append(download_audio(url, audio_path))

    await asyncio.gather(*tasks)


async def check_exists(urls: List[str], timeout: float = 5.0) -> List[str]:
    """Check all audiobooks in parallel and return valid URLs with progress tracking.

    Args:
        urls: List of URLs to check
        timeout: Maximum time in seconds to wait for each request before considering it failed

    Returns:
        List of URLs that exist (return 200 status code)
    """

    async def check_a_book(name: str, session: aiohttp.ClientSession) -> bool:
        """Check if a book exists using aiohttp, handling timeout errors."""
        try:
            async with session.head(
                url=constants.TEXT_BASE_URL + name, timeout=timeout
            ) as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            # Handle both client errors and timeouts
            return False

    print("Checking the existence of the book in the text source")
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = [check_a_book(url.split("/")[-1], session) for url in urls]

        # Use tqdm_asyncio to track progress
        results = await tqdm_asyncio.gather(*tasks, desc="Checking URLs")

    # Filter valid URLs
    return [url for url, exist in zip(urls, results) if exist]


async def get_metadata_batch(
    batch: List[Tuple[str, str]], semaphore: asyncio.Semaphore
):
    """Process a batch of metadata with rate limiting."""
    async with semaphore:
        results = []
        for text_url, audio_url in batch:
            result = await utils.get_metadata(text_url, audio_url, save=True)
            results.append(result)
        return results


async def main():
    """
    Main function to get all audiobook URLs and download them.
    """
    os.makedirs(constants.AUDIO_SAVEPATH, exist_ok=True)
    os.makedirs(
        constants.TEXT_SAVEPATH, exist_ok=True
    )  # Added to ensure both dirs exist

    # Get all audiobook URLs
    print("Getting all audiobook URLs and names")
    all_audio_urls = await utils.audio.get_all_audiobook_url()
    print(f"Found {len(all_audio_urls)} audiobooks")

    # Save all audiobook's URLs
    print(f"Saving all audiobook URLs to {constants.ALL_AUDIOBOOK_URLS_SAVEPATH} file")
    async with aiofiles.open(constants.ALL_AUDIOBOOK_URLS_SAVEPATH, "w") as f:
        await f.write("\n".join(all_audio_urls))  # Optimized to write all at once

    # Filter books that exist in text source
    valid_audio_urls = await check_exists(
        all_audio_urls
    )  # Using the optimized check_exists function
    print(
        f"After checking the existence of the book in the text source, "
        f"found {len(valid_audio_urls)} audiobooks to download"
    )

    # Get metadata for each book
    print(
        f"Getting metadata for each book and save it to JSON file in {constants.METADATA_SAVEPATH}"
    )
    text_urls = [
        constants.TEXT_BASE_URL + url.split("/")[-1] for url in valid_audio_urls
    ]

    # Create pairs of URLs and use a semaphore to limit concurrent metadata fetches
    url_pairs = list(zip(text_urls, valid_audio_urls))

    # Use a semaphore to limit concurrency for metadata fetching
    concurrency_limit = min(10, len(url_pairs))  # Adjust based on API limits
    semaphore = asyncio.Semaphore(concurrency_limit)

    # Process metadata in smaller batches
    batch_size = 5  # Adjust based on performance
    batches = [
        url_pairs[i : i + batch_size] for i in range(0, len(url_pairs), batch_size)
    ]

    metadata_tasks = [get_metadata_batch(batch, semaphore) for batch in batches]

    for batch_idx, task in enumerate(
        tqdm(
            asyncio.as_completed(metadata_tasks),
            total=len(metadata_tasks),
            desc="Processing metadata batches",
        )
    ):
        await task

    # Download books with limited concurrency
    print("Downloading books concurrently")
    download_semaphore = asyncio.Semaphore(5)  # Limit concurrent downloads

    async def download_with_semaphore(audio_url, text_url):
        async with download_semaphore:
            return await download_full_book(
                audio_url, text_url, constants.AUDIO_SAVEPATH, constants.TEXT_SAVEPATH
            )

    download_tasks = [
        download_with_semaphore(audio_url, text_url)
        for audio_url, text_url in zip(valid_audio_urls, text_urls)
    ]

    for completed_task in tqdm(
        asyncio.as_completed(download_tasks),
        total=len(download_tasks),
        desc="Downloading books",
    ):
        await completed_task

    print("Download complete!")


if __name__ == "__main__":
    asyncio.run(main())
