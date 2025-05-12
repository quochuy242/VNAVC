import argparse
import asyncio
import os
import os.path as osp

import aiofiles
from tqdm.asyncio import tqdm

from tts_data_pipeline import constants

from tts_data_pipeline.crawler import download, metadata, utils
from tts_data_pipeline.crawler.utils import logger


def parse_args():
  parser = argparse.ArgumentParser(description="Audiobook Download Pipeline")
  parser.add_argument(
    "-s",
    "--save-urls",
    action="store_true",
    help="Force to save all audiobook URLs to a file",
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
    help="Download books by name when --download is query",
    default=None,
  )
  parser.add_argument(
    "--author",
    type=str,
    help="Download books by author when --download is query",
    default=None,
  )
  parser.add_argument(
    "--narrator",
    type=str,
    help="Download books by narrator when --download is query",
    default=None,
  )
  return parser.parse_args()


async def main():
  """
  Main function to get all audiobook URLs and download them.
  """
  args = parse_args()

  # Added to ensure both dirs exist
  os.makedirs(constants.AUDIO_SAVE_PATH, exist_ok=True)
  os.makedirs(constants.TEXT_SAVE_PATH, exist_ok=True)

  # Get all book's URLs
  if args.save_urls or not osp.exists(constants.ALL_AUDIOBOOK_URLS_SAVE_PATH):
    logger.info("Getting all audiobook URLs and names")
    audio_urls = await utils.get_all_audiobook_url()
    logger.info(f"Found {len(audio_urls)} audiobooks")
  else:
    logger.info(
      f"Loading all audiobook URLs from {constants.ALL_AUDIOBOOK_URLS_SAVE_PATH} file"
    )
    async with aiofiles.open(constants.ALL_AUDIOBOOK_URLS_SAVE_PATH, "r") as f:
      audio_urls = (await f.read()).splitlines()

  if args.save_urls:
    logger.info(
      f"Saving all audiobook URLs to {constants.ALL_AUDIOBOOK_URLS_SAVE_PATH} file"
    )
    async with aiofiles.open(constants.ALL_AUDIOBOOK_URLS_SAVE_PATH, "w") as f:
      await f.write("\n".join(audio_urls))  # Optimized to write all at once
  else:
    logger.warning("Not saving all audiobook URLs to text file")

  # Fetch metadata
  text_urls = [f"{constants.TEXT_BASE_URL}{url.split('/')[-1]}" for url in audio_urls]
  if args.fetch_metadata:
    await metadata.fetch_metadata(text_urls, audio_urls)

  # Create metadata CSV
  if args.create_metadata_csv:
    logger.info("Process and convert metadata files to a single CSV file")
    await asyncio.to_thread(metadata.convert_metadata_to_csv)

  # Download books with limited concurrency
  if args.download.lower() == "all":
    logger.info("Downloading all books")
    valid_audio_urls = await asyncio.to_thread(
      metadata.get_valid_audio_urls,
      query="all",
      name=None,
      author=None,
      narrator=None,
    )
  elif args.download.lower() == "query":
    logger.info("Downloading books by query")
    valid_audio_urls = await asyncio.to_thread(
      metadata.get_valid_audio_urls,
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

  text_download_urls = [
    await utils.get_text_download_url(url.split("/")[-1]) for url in valid_audio_urls
  ]
  logger.info("Downloading books concurrently")
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
  logger.success("Download complete!")


if __name__ == "__main__":
  asyncio.run(main())
