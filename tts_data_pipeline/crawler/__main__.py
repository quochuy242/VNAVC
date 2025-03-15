import asyncio
import os
import argparse

import aiofiles
from rich import print
from tqdm.asyncio import tqdm

from tts_data_pipeline import constants

from . import download, metadata, utils


def parse_args():
    parser = argparse.ArgumentParser(description="Audiobook Download Pipeline")
    parser.add_argument(
        "--save-urls",
        action="store_true",
        help="Saving all audiobook URLs to a file",
    )
    parser.add_argument(
        "--fetch-metadata", action="store_true", help="Fetch metadata for each book"
    )
    parser.add_argument(
        "--convert-metadata", action="store_true", help="Convert metadata to CSV"
    )
    parser.add_argument("--download-books", action="store_true", help="Download books")
    return parser.parse_args()


async def main():
    """
    Main function to get all audiobook URLs and download them.
    """
    args = parse_args()

    # Added to ensure both dirs exist
    os.makedirs(constants.AUDIO_SAVE_PATH, exist_ok=True)
    os.makedirs(constants.TEXT_SAVE_PATH, exist_ok=True)

    # Get all audiobook URLs
    if not os.path.exists(constants.ALL_AUDIOBOOK_URLS_SAVE_PATH):
        print("Getting all audiobook URLs and names")
        audio_urls = await utils.audio.get_all_audiobook_url()
        print(f"Found {len(audio_urls)} audiobooks")

        # Save all audiobook's URLs
        if args.save_all_urls:
            print(
                f"Saving all audiobook URLs to {constants.ALL_AUDIOBOOK_URLS_SAVE_PATH} file"
            )
            async with aiofiles.open(constants.ALL_AUDIOBOOK_URLS_SAVE_PATH, "w") as f:
                await f.write("\n".join(audio_urls))  # Optimized to write all at once
        else:
            print("Not saving all audiobook URLs to text file")
    else:
        print(
            f"Loading all audiobook URLs from {constants.ALL_AUDIOBOOK_URLS_SAVE_PATH} file"
        )
        async with aiofiles.open(constants.ALL_AUDIOBOOK_URLS_SAVE_PATH, "r") as f:
            audio_urls = (await f.read()).splitlines()

    # Get metadata for each book
    if args.fetch_metadata:
        text_urls = [
            f"{constants.TEXT_BASE_URL}{url.split('/')[-1]}" for url in audio_urls
        ]
        print("Getting metadata for each book")
        print(f"Save it to JSON file in {constants.METADATA_SAVE_PATH}")
        fetch_metadata_limit = min(
            constants.FETCH_METADATA_LIMIT, len(text_urls)
        )  # Use a semaphore to limit concurrency for metadata fetching
        semaphore = asyncio.Semaphore(fetch_metadata_limit)

        metadata_tasks = [
            metadata.get_metadata(
                text_url, audio_url, semaphore, constants.METADATA_SAVE_PATH
            )
            for text_url, audio_url in zip(text_urls, audio_urls)
        ]
        for task in tqdm(
            asyncio.as_completed(metadata_tasks),
            total=len(metadata_tasks),
            desc="Fetching metadata",
        ):
            await task

    # Prepare book metadata
    if args.convert_metadata:
        print("Converting metadata to CSV")
        await asyncio.to_thread(metadata.convert_metadata_to_csv)
    elif not os.path.exists(constants.METADATA_BOOK_PATH):
        print(
            f"Metadata CSV file not found at {constants.METADATA_BOOK_PATH}. "
            "You should run the script with --convert-metadata option."
        )

    if args.download_books:
        # Get text URLs from valid audio URLs
        valid_audio_urls = await asyncio.to_thread(metadata.get_valid_audio_urls)
        text_download_urls = [
            await utils.get_text_download_url(url.split("/")[-1])
            for url in valid_audio_urls
        ]

        # Download books with limited concurrency
        print("Downloading books concurrently")
        download_semaphore = asyncio.Semaphore(
            constants.DOWNLOAD_BOOK_LIMIT
        )  # Limit concurrent downloads
        download_tasks = [
            download.download_with_semaphore(
                audio_url,
                text_url,
                audio_save_path=constants.AUDIO_RAW_DIR,
                text_save_path=constants.TEXT_PDF_DIR,
                download_semaphore=download_semaphore,
            )
            for audio_url, text_url in zip(valid_audio_urls, text_download_urls)
        ]
        for task in tqdm(
            asyncio.as_completed(download_tasks),
            total=len(download_tasks),
            desc="Downloading books",
        ):
            await task
        print("Download complete!")
    else:
        print("Not downloading books")


if __name__ == "__main__":
    asyncio.run(main())
