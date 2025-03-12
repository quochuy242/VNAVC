import asyncio
import os

import aiofiles
from rich import print
from tqdm.asyncio import tqdm

from tts_data_pipeline import constants

from tts_data_pipeline.crawler import utils


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
        all_audio_urls = await utils.audio.get_all_audiobook_url()
        print(f"Found {len(all_audio_urls)} audiobooks")

        # Save all audiobook's URLs
        print(
            f"Saving all audiobook URLs to {constants.ALL_AUDIOBOOK_URLS_SAVE_PATH} file"
        )
        async with aiofiles.open(constants.ALL_AUDIOBOOK_URLS_SAVE_PATH, "w") as f:
            await f.write("\n".join(all_audio_urls))  # Optimized to write all at once
    else:
        print(
            f"Loading all audiobook URLs from {constants.ALL_AUDIOBOOK_URLS_SAVE_PATH} file"
        )
        async with aiofiles.open(constants.ALL_AUDIOBOOK_URLS_SAVE_PATH, "r") as f:
            all_audio_urls = (await f.read()).splitlines()

    # Filter books that exist in text source
    # if not os.path.exists(constants.TEXT_BOOK_URLS_SAVE_PATH):
    #     valid_audio_urls = await utils.check_exists(
    #         all_audio_urls
    #     )  # Using the optimized check_exists function
    #     print(
    #         f"After checking the existence of the book in the text source, "
    #         f"found {len(valid_audio_urls)} audiobooks to download"
    #     )
    # else:
    #     print(
    #         f"Loading valid audiobook URLs from {constants.TEXT_BOOK_URLS_SAVE_PATH} file"
    #     )
    #     async with aiofiles.open(constants.TEXT_BOOK_URLS_SAVE_PATH, "r") as f:
    #         valid_audio_urls = (await f.read()).splitlines()

    valid_audio_urls = all_audio_urls

    # Get text URLs from valid audio URLs
    text_urls = [
        constants.TEXT_DOWNLOAD_URL(url.split("/")[-1]) for url in valid_audio_urls
    ]

    # # Get metadata for each book
    # print(
    #     f"Getting metadata for each book and save it to JSON file in {constants.METADATA_SAVE_PATH}"
    # )

    # ## Use a semaphore to limit concurrency for metadata fetching
    # fetch_metadata_limit = min(
    #     constants.FETCH_METADATA_LIMIT, len(text_urls)
    # )  # Adjust based on API limits
    # semaphore = asyncio.Semaphore(fetch_metadata_limit)

    # ## Process metadata in smaller batches
    # metadata_tasks = [
    #     utils.get_metadata(text_url, audio_url, semaphore, constants.METADATA_SAVE_PATH)
    #     for text_url, audio_url in zip(text_urls, valid_audio_urls)
    # ]
    # for task in tqdm(
    #     asyncio.as_completed(metadata_tasks),
    #     total=len(metadata_tasks),
    #     desc="Processing metadata batches",
    # ):
    #     await task

    # Download books with limited concurrency
    print("Downloading books concurrently")
    download_semaphore = asyncio.Semaphore(
        constants.DOWNLOAD_BOOK_LIMIT
    )  # Limit concurrent downloads

    download_tasks = [
        utils.download_with_semaphore(audio_url, text_url, download_semaphore)
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
