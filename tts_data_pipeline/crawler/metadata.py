import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

import httpx
import pandas as pd
from tqdm.asyncio import tqdm
import requests
import io
import csv

from tts_data_pipeline import constants, Book, Narrator, convert_duration
from tts_data_pipeline.crawler import utils
from tts_data_pipeline.crawler.utils import logger

async def get_book_metadata(
  text_url: str, audio_url: str, semaphore: asyncio.Semaphore, save_path: str = ""
) -> Optional[Book]:
  """
  Asynchronously get book metadata and return a Book instance.

  Args:
      text_url (str): The URL of the book page.
      audio_url (str): The URL of the audiobook page.
      semaphore (asyncio.Semaphore): Concurrency control.
      save_path (str): Folder path to save metadata JSON (optional).

  Returns:
      Optional[Book]: A Book instance or None if an error occurred.
  """
  async with semaphore:
    try:
      text_parser = await utils.get_web_content(text_url)
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

    book = Book(
      name=title,
      author=author,
      duration=duration,
      narrator=narrators if len(narrators) > 1 else narrators[0],
      text_url=text_url,
      audio_url=audio_url,
    )

    if save_path:
      os.makedirs(save_path, exist_ok=True)
      file_name = f"{text_url.split('/')[-1]}.json"
      full_path = Path(save_path) / file_name
      book.save_json(full_path)
    else:
      logger.info("Don't save any book's metadata")

    return book


async def fetch_book_metadata(text_urls: List[str], audio_urls: List[str]):
  logger.info("Fetching metadata for each book")
  logger.info(f"Save it to JSON file in {constants.METADATA_SAVE_PATH}")
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

    # Convert duration to hours
    df["duration_hour"] = df["duration"].apply(convert_duration, unit="hour")

    # Add new columns
    df["sample_rate"] = pd.Series([None] * len(df))
    df["quality"] = pd.Series([None] * len(df))
    df["word_count"] = pd.Series([None] * len(df))
    df["num_sentences"] = pd.Series([None] * len(df))

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


def get_narrator_metadata():
  """
  Get metadata for each narrator from google sheet file.
  """
  # os.makedirs(os.path.dirname(constants.METADATA_NARRATOR_PATH), exist_ok=True)
  try:
    # Download data directly from GG Sheet
    headers = {
      "User-Agent": "Mozilla/5.0",
      "Accept": "text/csv"
    }
    response = requests.get(
      constants.METADATA_DOWNLOAD_URL, 
      headers=headers,
      allow_redirects=True
    )
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Check content-type
    if "text/html" in response.headers.get("content-type", "").lower():
      raise ValueError("Received HTML instead of CSV")

    # Read CSV from response content
    df = pd.read_csv(
      io.StringIO(response.content.decode("utf-8")),
      dtype=str,              # ensure all columns are read as strings  
      keep_default_na=False  # treat empty cells as NaN
    )
    return df

  except Exception as e:
    logger.error(f"Error: Can't narrator metadata: {str(e)}")
    return pd.DataFrame()

  # try:
  #   # Read directly from content with encoding and quoting
  #   df = pd.read_csv(
  #     constants.METADATA_DOWNLOAD_URL,
  #     encoding="utf-8",
  #     dtype=str,              # ensure all columns are read as strings
  #     keep_default_na=False,  # treat empty cells as NaN
  #     on_bad_lines="skip",    # skip bad lines
  #     quoting=csv.QUOTE_ALL,  # handle all quoting
  #     escapechar="\\",        # except escape character
  #     engine="python",        # use engine python instead of C
  #   )

  #   # Clean up column names
  #   df.columns = df.columns.str.strip().str.lower()

  #   # Drop empty rows
  #   df.dropna(how="all")
    
  #   return df

  # except Exception as e:
  #     logger.error(f"Error: Can't read csv file: {str(e)}")
  #     return pd.DataFrame()   # Return an empty DataFrame if there is an error
  
  # def convert_csv_to_json(df: pd.DataFrame) -> List[dict]:
  #     """
  #     Convert a DataFrame to a list of json file.
  #     """
  #     # Convert DataFrame to JSON
  #     json_data = df.to_dict(orient="records")
      
  #     # Write to JSON file with proper encoding and indentation
  #     with open(constants.METADATA_NARRATOR_PATH, "w", encoding="utf-8") as f:
  #         json.dump(json_data, f, ensure_ascii=False, indent=2)

  #     return json_data

  # json_data = convert_csv_to_json(df)
  # return json_data

# return metadata


def convert_duration(time_str: str, unit: str = "second") -> float | None:
    """
    Convert a time string in the format "HH:MM:SS" or "MM:SS" to the specified unit (seconds, minutes, or hours).
    """
    if not isinstance(time_str, str):
        return None

    try:
        time_values = time_str.split(":")
        total_seconds = sum(
            int(num) * 60**i for i, num in enumerate(reversed(time_values))
        )
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
    metadata_path = Path(constants.METADATA_SAVE_PATH)

    # Create the output directory if it doesn't exist
    metadata_path.mkdir(parents=True, exist_ok=True)

    # Get all JSON files from the metadata directory
    json_files = metadata_path.glob("*.json")
    all_metadata = []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                # Load JSON data
                data = json.load(f)

                # Convert duration to hours
                if "duration" in data and isinstance(data["duration"], str):
                    data["duration_hours"] = convert_duration(data["duration"], "hour")

                # Append to the list
                all_metadata.append(data)
            except json.JSONDecodeError:
                print(f"Error parsing JSON file: {json_file}")

    # Convert to DataFrame
    if all_metadata:
        df = pd.DataFrame(all_metadata)

        # Save the combined metadata as CSV
        df.to_csv(constants.METADATA_BOOK_PATH, index=True)

        print(
            f"Metadata processing complete. {len(all_metadata)} files processed. Saved to {constants.METADATA_BOOK_PATH}"
        )
    else:
        print("No metadata files were processed.")


def get_valid_audio_urls() -> List[str]:
    """
    Get a list of valid audio URLs from the metadata CSV file.
    """
    return pd.read_csv(constants.METADATA_BOOK_PATH)["audio_url"].tolist()

#  convert all metadata json to a single file csv, so I have the valid download URL audio. The downloading progress will be completed soon)
