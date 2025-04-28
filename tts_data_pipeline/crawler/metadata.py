import asyncio
import json
import os
import random as randomlib
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import pandas as pd
from tqdm.asyncio import tqdm

from tts_data_pipeline import constants
from tts_data_pipeline.crawler import utils
from tts_data_pipeline.crawler.utils import logger


async def get_metadata(
  text_url: str, audio_url: str, semaphore: asyncio.Semaphore, save_path: str = None
) -> Dict[str, str] | None:
  """
  Asynchronously get audio metadata from an book URL.

  Args:
      text_url (str): The URL of the book page.
      audio_url (str): The URL of the audiobook page.
      save (bool): Whether to save the metadata as a JSON file.

  Returns:
      Dict[str, str]: The book metadata, containing title, url, duration, author, and narrator's name.
  """
  async with semaphore:
    try:
      text_parser = await utils.get_web_content(text_url)
      audio_parser = await utils.get_web_content(audio_url)
    except httpx.HTTPStatusError:
      return

    title = text_parser.css_first("h1.title-detail")
    author = text_parser.css_first(
      "div.product-price span.text-brand"
    )  # The text source is more reliable than audio one
    duration = audio_parser.css_first(".featu")
    narrator = audio_parser.css_first("i.fa-microphone + a")

    metadata = {
      "audio_url": audio_url,
      "text_url": text_url,
      "title": title.text(strip=True) if title else "Unknown",
      "author": author.text(strip=True) if author else "Unknown",
      "duration": duration.text(strip=True) if duration else "Unknown",
      "narrator": narrator.text(strip=True) if narrator else "Unknown",
    }

    if save_path:
      os.makedirs(save_path, exist_ok=True)
      save_path += f"{text_url.split('/')[-1]}.json"
      with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    else:
      print("Don't save any book's metadata")

    return metadata


async def fetch_metadata(text_urls: List[str], audio_urls: List[str]):
  logger.info("Fetching metadata for each book")
  logger.info(f"Save it to JSON file in {constants.METADATA_SAVE_PATH}")
  fetch_metadata_limit = min(
    constants.FETCH_METADATA_LIMIT, len(text_urls)
  )  # Use a semaphore to limit concurrency for metadata fetching
  semaphore = asyncio.Semaphore(fetch_metadata_limit)

  metadata_tasks = [
    get_metadata(text_url, audio_url, semaphore, constants.METADATA_SAVE_PATH)
    for text_url, audio_url in zip(text_urls, audio_urls)
  ]
  for task in tqdm(
    asyncio.as_completed(metadata_tasks),
    total=len(metadata_tasks),
    desc="Fetching metadata",
  ):
    await task


def convert_duration(time_str: str, unit: str = "second") -> float | None:
  """
  Convert a time string in the format "HH:MM:SS" or "MM:SS" to the specified unit (seconds, minutes, or hours).
  """
  if not isinstance(time_str, str):
    return None

  try:
    time_values = time_str.split(":")
    total_seconds = sum(int(num) * 60**i for i, num in enumerate(reversed(time_values)))

    match unit.lower():
      case "second":
        return total_seconds
      case "minute":
        return round(total_seconds / 60, 4)
      case "hour":
        return round(total_seconds / 3600, 4)
      case _:
        return None  # Invalid unit
  except ValueError:
    return None


def convert_metadata_to_csv():
  """
  Reads JSON metadata files, saves all metadata to a single file as CSV.
  """

  def process_df(df: pd.DataFrame) -> pd.DataFrame:
    # Convert duration to hours
    df["duration_hour"] = df["duration"].apply(convert_duration, unit="hour")
    # Remove the tvshow
    df = df[~df["audio_url"].str.contains("tvshows", na=False)]
    # Special cases
    df.loc[217, "duration"] = None  # Dang cap nhat
    df.loc[536, "duration"] = "4:05:10"  # 4:05;10
    df.loc[716, "duration"] = "8:14:52"  # 8:14::52
    df.loc[1046, "duration"] = "2:01:40"  # 2:01;40
    df.loc[1173, "duration"] = "25:21:20"  # 25:21:20:
    df.loc[1197, "duration"] = "13:23:03"  # 13;23;03
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
        print(f"Error parsing JSON file: {json_file}")

  # Convert to DataFrame
  if all_metadata:
    df = pd.DataFrame(all_metadata)
    df = process_df(df)

    # Save the combined metadata as CSV
    df.to_csv(constants.METADATA_BOOK_PATH, index=False)

    print(
      f"Metadata processing complete. {len(all_metadata)} files processed. Saved to {constants.METADATA_BOOK_PATH}"
    )
  else:
    print("No metadata files were processed.")


def get_valid_audio_urls(
  query: Optional[str],
  name: Optional[str],
  author: Optional[str],
  narrator: Optional[str],
  random: int = 0,
) -> List[str]:
  """
  Get a list of valid audio URLs from the metadata CSV file.
  """
  df = pd.read_csv(constants.METADATA_BOOK_PATH)

  if random > 0:
    return randomlib.sample(df["audio_url"].tolist(), random)

  if query == "all":
    return df["audio_url"].tolist()
  else:
    mask = pd.Series([True] * len(df))
    if name:
      mask &= df["title"].str.contains(name, na=False)
    if author:
      mask &= df["author"].str.contains(author, na=False)
    if narrator:
      mask &= df["narrator"].str.contains(narrator, na=False)
    return df[mask]["audio_url"].tolist()
