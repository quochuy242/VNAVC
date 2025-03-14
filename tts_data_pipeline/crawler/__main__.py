import asyncio
import os

import aiofiles
from rich import print
from tqdm.asyncio import tqdm

from tts_data_pipeline import constants

from . import download, metadata, utils


async def main():
    """
    Main function to get all audiobook URLs and download them.
    """
    os.makedirs(constants.AUDIO_SAVE_PATH, exist_ok=True)
    os.makedirs(
        constants.TEXT_SAVE_PATH, exist_ok=True
    )  # Added to ensure both dirs exist

    # Get all audiobook URLs
    if not os.path.exists(constants.ALL_AUDIOBOOK_URLS_SAVE_PATH):
        print("Getting all audiobook URLs and names")
        audio_urls = await utils.audio.get_all_audiobook_url()
        print(f"Found {len(audio_urls)} audiobooks")

        # Save all audiobook's URLs
        print(
            f"Saving all audiobook URLs to {constants.ALL_AUDIOBOOK_URLS_SAVE_PATH} file"
        )
        async with aiofiles.open(constants.ALL_AUDIOBOOK_URLS_SAVE_PATH, "w") as f:
            await f.write("\n".join(audio_urls))  # Optimized to write all at once
    else:
        print(
            f"Loading all audiobook URLs from {constants.ALL_AUDIOBOOK_URLS_SAVE_PATH} file"
        )
        async with aiofiles.open(constants.ALL_AUDIOBOOK_URLS_SAVE_PATH, "r") as f:
            audio_urls = (await f.read()).splitlines()

    # Get text URLs from valid audio URLs
    text_download_urls = [
        await utils.get_text_download_url(url.split("/")[-1]) for url in audio_urls
    ]

    # Get metadata for each book
    # text_urls = [f"{constants.TEXT_BASE_URL}{url.split('/')[-1]}" for url in audio_urls]
    # print(
    #     f"Getting metadata for each book and save it to JSON file in {constants.METADATA_SAVE_PATH}"
    # )
    # fetch_metadata_limit = min(
    #     constants.FETCH_METADATA_LIMIT, len(text_urls)
    # )  # Use a semaphore to limit concurrency for metadata fetching
    # semaphore = asyncio.Semaphore(fetch_metadata_limit)

    # metadata_tasks = [
    #     metadata.get_metadata(
    #         text_url, audio_url, semaphore, constants.METADATA_SAVE_PATH
    #     )
    #     for text_url, audio_url in zip(text_urls, audio_urls)
    # ]
    # for task in tqdm(
    #     asyncio.as_completed(metadata_tasks),
    #     total=len(metadata_tasks),
    #     desc="Fetching metadata",
    # ):
    #     await task

    # Download books with limited concurrency
    print("Downloading books concurrently")
    download_semaphore = asyncio.Semaphore(
        constants.DOWNLOAD_BOOK_LIMIT
    )  # Limit concurrent downloads
    download_tasks = [
        download.download_with_semaphore(
            audio_url,
            text_url,
            audio_save_path=constants.RAW_DIR,
            text_save_path=constants.PDF_DIR,
            download_semaphore=download_semaphore,
        )
        for audio_url, text_url in zip(audio_urls, text_download_urls)
    ]
    for task in tqdm(
        asyncio.as_completed(download_tasks),
        total=len(download_tasks),
        desc="Downloading books",
    ):
        await task
    print("Download complete!")


if __name__ == "__main__":
    asyncio.run(main())
