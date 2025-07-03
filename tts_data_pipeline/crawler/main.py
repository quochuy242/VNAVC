import argparse
import asyncio
import os
import os.path as osp

import aiofiles
import pandas as pd

from tts_data_pipeline import constants
from tts_data_pipeline.crawler import download, metadata, utils
from tts_data_pipeline.crawler.utils import logger


def parse_args():
  parser = argparse.ArgumentParser(description="Audiobook Download Pipeline")
  parser.add_argument(
    "--check-urls",
    action="store_true",
    help="Force to check the existance of all both text and audiobook URLs",
  )
  parser.add_argument(
    "-f",
    "--fetch-metadata",
    action="store_true",
    help="Force to fetch metadata for each book",
  )
  parser.add_argument(
    "--create-metadata-csv",
    action="store_true",
    help="Process and convert metadata files to a single CSV file",
  )
  parser.add_argument(
    "-d",
    "--download",
    type=str,
    default="none",
    help="Download books (available: all, none, query) (default: none)",
  )
  parser.add_argument(
    "--name",
    type=str,
    help="Download books by name when --download is query, check the metadata file for exact match",
    default=None,
  )
  parser.add_argument(
    "--author",
    type=str,
    help="Download books by author when --download is query, check the metadata file for exact match",
    default=None,
  )
  parser.add_argument(
    "--narrator",
    type=str,
    help="Download books by narrator when --download is query, check the metadata file for exact match",
    default=None,
  )
  return parser.parse_args()


async def load_urls():
  """
  Load audio and text URLs from files.
  """
  audio_urls = []
  text_urls = []
  async with aiofiles.open(constants.VALID_BOOK_URL_SAVE_PATH, "r") as f:
    async for line in f:
      audio_url, text_url, source = line.strip().split(", ")
      audio_urls.append(audio_url)
      text_urls.append((text_url, source))
  return audio_urls, text_urls


async def main():
  """
  Main function to get all audiobook URLs and download them.
  """
  args = parse_args()

  # Added to ensure both dirs exist
  os.makedirs(constants.AUDIO_RAW_DIR, exist_ok=True)
  os.makedirs(constants.TEXT_PDF_DIR, exist_ok=True)

  # Get all book's URLs
  if args.check_urls or not osp.exists(constants.VALID_BOOK_URL_SAVE_PATH):
    logger.info("Getting all audiobook URLs and names")
    text_audio_urls: pd.DataFrame = await utils.get_all_book_url()
    logger.info(
      f"Found {len(text_audio_urls[text_audio_urls['source'] == 'invalid'])} ({len(text_audio_urls[text_audio_urls['source'] == 'invalid']) / len(text_audio_urls) * 100:.2f}%) invalid audiobooks"
    )

    # Save URLs to files
    async with aiofiles.open(constants.VALID_BOOK_URL_SAVE_PATH, "w") as f:
      valid_urls = [
        f"{row['audio_url']}, {row['text_url']}, {row['source']}"
        for _, row in text_audio_urls.iterrows()
      ]
      await f.write("\n".join(valid_urls))

  # Fetch metadata
  if args.fetch_metadata:
    logger.info(f"Loading all audiobook URLs from {constants.VALID_BOOK_URL_SAVE_PATH}")
    audio_urls, text_urls = await load_urls()
    await metadata.fetch_book_metadata(text_urls, audio_urls)

  # Create metadata CSV
  if args.create_metadata_csv:
    logger.info("Process and convert metadata files to a single CSV file")
    await asyncio.to_thread(metadata.convert_metadata_to_csv)

  # Download books with limited concurrency
  if args.download.lower() == "all":
    logger.info("Downloading all books")
    audio_download_urls, text_download_urls = await asyncio.to_thread(
      utils.query_download_url,
      query="all",
    )
  elif args.download.lower() == "query":
    logger.info("Downloading books by query")
    audio_download_urls, text_download_urls = await asyncio.to_thread(
      utils.query_download_url,
      query="query",
      name=args.name,
      author=args.author,
      narrator=args.narrator,
    )
  elif args.download.lower() == "none":
    logger.info("Not downloading books")
    return
  else:
    logger.error("Invalid download option")
    return

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
    for audio_url, text_url in zip(audio_download_urls, text_download_urls)
  ]
  for task in asyncio.as_completed(download_tasks):
    await task

  # Group audiobook task
  try:
    await asyncio.to_thread(
      utils.group_audiobook, constants.AUDIO_RAW_DIR, constants.AUDIO_UNQUALIFIED_DIR
    )
  except Exception as e:
    logger.exception(f"Error grouping audiobooks: {e}")


if __name__ == "__main__":
  asyncio.run(main())
