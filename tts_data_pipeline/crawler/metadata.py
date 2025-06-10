import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import pandas as pd
from tqdm.asyncio import tqdm

from tts_data_pipeline import constants, Book, Narrator
from tts_data_pipeline.crawler import utils
from tts_data_pipeline.crawler.utils import (
  logger,
  get_text_download_url,
  fetch_download_audio_url,
)


async def get_book_metadata(
  text_url: Tuple[str, str],
  audio_url: str,
  semaphore: asyncio.Semaphore,
  save_path: str = "",
) -> Optional[Book]:
  """
  Asynchronously get book metadata and return a Book instance.

  Args:
      text_url (Tuple[str, str]): The URL of the book page and the source's name.
      audio_url (str): The URL of the audiobook page.
      semaphore (asyncio.Semaphore): Concurrency control.
      save_path (str): Folder path to save metadata JSON (optional).

  Returns:
      Optional[Book]: A Book instance or None if an error occurred.
  """
  async with semaphore:
    try:
      text_parser = await utils.get_web_content(text_url[0])
      audio_parser = await utils.get_web_content(audio_url)
    except httpx.HTTPStatusError:
      return None

    # Extract fields from HTML
    title_tag = text_parser.css_first("h1.title-detail")
    author_tag = text_parser.css_first("div.product-price span.text-brand")
    duration_tag = audio_parser.css_first(".featu")

    title = title_tag.text(strip=True) if title_tag else "Unknown"
    author = author_tag.text(strip=True) if author_tag else "Unknown"
    duration = duration_tag.text(strip=True) if duration_tag else "Unknown"

    # Parse narrators
    narrator_tags = audio_parser.css("i.fa-microphone ~ a")
    narrators: List[Narrator] = []

    for tag in narrator_tags:
      name = tag.text(strip=True) or "Unknown"
      url = tag.attributes.get("href", "Unknown")
      narrators.append(Narrator(name=name, url=url))

    if not narrators:
      narrators.append(Narrator(name="Unknown", url="Unknown"))

    if text_url[1] != "invalid":
      audio_download_url = await fetch_download_audio_url(audio_url)
      text_download_url = await get_text_download_url(
        text_url[0].split("/")[-1], source=text_url[1]
      )
    else:
      audio_download_url = None
      text_download_url = None

    book = Book(
      name=title,
      author=author,
      duration=duration,
      narrator=narrators if len(narrators) > 1 else narrators[0],
      text_url=text_url,
      audio_url=audio_url,
      text_download_url=text_download_url,
      audio_download_url=audio_download_url,
    )

    if save_path:
      os.makedirs(save_path, exist_ok=True)
      file_name = f"{text_url[0].split('/')[-1]}.json"
      full_path = Path(save_path) / file_name
      book.save_json(full_path)
    else:
      logger.info("Don't save any book's metadata")

    return book


async def fetch_book_metadata(text_urls: List[Tuple[str, str]], audio_urls: List[str]):
  logger.info(
    f"Fetching metadata for each book, save it to JSON file in {constants.METADATA_SAVE_PATH}"
  )
  fetch_metadata_limit = min(
    constants.FETCH_METADATA_LIMIT, len(text_urls)
  )  # Use a semaphore to limit concurrency for metadata fetching
  semaphore = asyncio.Semaphore(fetch_metadata_limit)

  metadata_tasks = [
    get_book_metadata(text_url, audio_url, semaphore, constants.METADATA_SAVE_PATH)
    for text_url, audio_url in zip(text_urls, audio_urls)
  ]
  for task in tqdm(
    asyncio.as_completed(metadata_tasks),
    total=len(metadata_tasks),
    desc="Fetching metadata",
  ):
    await task


def convert_metadata_to_csv():
  """
  Reads JSON metadata files, saves all metadata to a single file as CSV.
  """

  def process_df(df: pd.DataFrame) -> pd.DataFrame:
    # Remove the tvshow
    df = df[~df["audio_url"].str.contains("tvshows", na=False)].copy()

    # Add new columns
    df["sample_rate"] = pd.Series([None] * len(df))
    df["quality"] = pd.Series([None] * len(df))
    df["word_count"] = pd.Series([None] * len(df))
    df["num_sentences"] = pd.Series([None] * len(df))
    df["audio_size"] = pd.Series([None] * len(df))
    df["text_size"] = pd.Series([None] * len(df))

    return df

  metadata_path = Path(constants.METADATA_SAVE_PATH)

  # Create the output directory if it doesn't exist
  metadata_path.mkdir(parents=True, exist_ok=True)

  # Get all JSON files from the metadata directory
  json_files = metadata_path.glob("*.json")
  all_metadata = []

  for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
      try:
        data = json.load(f)
        all_metadata.append(data)
      except json.JSONDecodeError:
        logger.info(f"Error parsing JSON file: {json_file}")

  # Convert to DataFrame
  if all_metadata:
    df = pd.DataFrame(all_metadata)
    df = process_df(df)

    # Save to CSV
    df.to_csv(constants.METADATA_BOOK_PATH, index=False)
    logger.info(
      f"Metadata saved to {constants.METADATA_BOOK_PATH}. {len(all_metadata)} files processed"
    )
  else:
    logger.info("No metadata files were processed.")


# TODO: Get metadata for each narrator from google sheet file
def get_narrator_metadata():
  pass
