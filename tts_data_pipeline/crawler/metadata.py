import asyncio
import json
import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

import aiofiles
import httpx
import pandas as pd
import requests
import typer
from rich.logging import RichHandler
from tqdm.asyncio import tqdm

# Assuming Book and Narrator models are defined somewhere accessible.
from tts_data_pipeline import Book, Narrator
from tts_data_pipeline.crawler import utils
from tts_data_pipeline.crawler.utils import (
  logger,
  get_text_download_url,
  fetch_download_audio_url,
)

# Initialize Typer app
app = typer.Typer(help="Audiobook Metadata Pipeline")  # Changed description

# Configure rich logger
logger.add(
  RichHandler(
    markup=True,
    rich_tracebacks=True,
    tracebacks_show_locals=True,
  ),
  level="INFO",
)


class Config:
  def __init__(self, config_path: Path = Path("config.json")):
    self.config_path = config_path
    self._load_config()

  def _load_config(self):
    if not self.config_path.exists():
      self._create_default_config()
    with open(self.config_path, "r") as f:
      # Load and convert all keys to lowercase
      self.data = {k.lower(): v for k, v in json.load(f).items()}

  def _create_default_config(self):
    default_config = {
      "valid_book_urls_path": "data/valid_book_urls.txt",
      "audio_raw_dir": "data/audio/raw",  # Still keep directories for consistency
      "text_pdf_dir": "data/text/pdf",  # even if not actively used for download here
      "metadata_dir": "data/metadata",
      "metadata_book_path": "data/metadata.csv",
      "metadata_narrator_path": "data/narrator_metadata.csv",
      "narrator_download_url": None,
      "fetch_metadata_limit": 5,
    }
    self.config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(self.config_path, "w") as f:
      json.dump(default_config, f, indent=4)
    logger.info(f"Created default config file at {self.config_path}")

  def __getattr__(self, name):
    if name in self.data:
      value = self.data[name]
      if name.endswith(("_path", "_dir")):
        return Path(value)
      return value
    raise AttributeError(
      f"'{self.__class__.__name__}' object has no attribute '{name}'"
    )


config = Config()


async def load_urls() -> tuple[list[str], list[tuple[str, str]]]:
  """
  Load audio and text URLs from files.
  """
  audio_urls = []
  text_urls = []
  async with aiofiles.open(config.valid_book_urls_path, "r") as f:
    async for line in f:
      audio_url, text_url, source = line.strip().split(", ")
      audio_urls.append(audio_url)
      text_urls.append((text_url, source))
  return audio_urls, text_urls


async def get_book_metadata(
  text_url: Tuple[str, str],
  audio_url: str,
  semaphore: asyncio.Semaphore,
  save_path: Path,
) -> Optional[Book]:
  """
  Asynchronously get book metadata and return a Book instance.
  """
  async with semaphore:
    try:
      text_parser = await utils.get_web_content(text_url[0])
      audio_parser = await utils.get_web_content(audio_url)
    except httpx.HTTPStatusError:
      logger.warning(f"HTTP error fetching content for {text_url[0]} or {audio_url}")
      return None
    except Exception as e:
      logger.error(f"Error fetching web content for {text_url[0]} or {audio_url}: {e}")
      return None

    # Extract fields from HTML
    title_tag = audio_parser.css_first("div.data h1") if audio_parser else None
    author_tag = (
      text_parser.css_first("div.product-price span.text-brand")
      if text_parser and text_url[1] == "taisachhay"
      else text_parser.css_first("div.entry-content a[href*='tac-gia']")
      if text_parser and text_url[1] == "thuviensach"
      else None
    )
    duration_tag = audio_parser.css_first(".featu") if audio_parser else None

    title = title_tag.text(strip=True) if title_tag else "Unknown"
    author = author_tag.text(strip=True) if author_tag else "Unknown"
    duration = duration_tag.text(strip=True) if duration_tag else "Unknown"

    # Parse narrators
    narrator_tags = audio_parser.css("i.fa-microphone ~ a") if audio_parser else []
    narrators: List[Narrator] = []

    for tag in narrator_tags:
      name = tag.text(strip=True) or "Unknown"
      url = tag.attributes.get("href", "Unknown")
      narrators.append(Narrator(name=name, url=url))

    if not narrators:
      narrators.append(Narrator(name="Unknown", url="Unknown"))

    audio_download_url = None
    text_download_url = None
    if text_url[1] != "invalid":
      try:
        audio_download_url = await fetch_download_audio_url(audio_url)
        text_filename_hint = text_url[0].split("/")[-1].split(".")[0]
        text_download_url = await get_text_download_url(
          text_filename_hint, source=text_url[1]
        )
      except Exception as e:
        logger.warning(f"Failed to get download URLs for {title}: {e}")
    else:
      logger.info(f"Skipping download URL extraction for invalid source: {title}")

    narrator_data: Union[Narrator, List[Narrator]]
    if len(narrators) == 1:
      narrator_data = narrators[0]
    else:
      narrator_data = narrators

    book = Book(
      name=title,
      author=author,
      duration=duration,
      narrator=narrator_data,
      text_url=text_url[0],
      audio_url=audio_url,
      text_download_url=text_download_url,
      audio_download_url=audio_download_url,
    )

    if save_path:
      save_path.mkdir(parents=True, exist_ok=True)
      file_name = f"{text_url[0].split('/')[-1]}.json"
      full_path = save_path / file_name
      book.save_json(full_path)
    else:
      logger.info("Don't save any book's metadata because save_path is empty.")

    return book


async def fetch_book_metadata(
  text_urls: List[Tuple[str, str]], audio_urls: List[str], metadata_save_path: Path
):
  logger.info(
    f"Fetching metadata for each book, saving to JSON files in {metadata_save_path}"
  )
  fetch_metadata_limit = min(config.fetch_metadata_limit, len(text_urls))
  semaphore = asyncio.Semaphore(fetch_metadata_limit)

  metadata_tasks = [
    get_book_metadata(text_url, audio_url, semaphore, metadata_save_path)
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
        return float(total_seconds)
      case "minute":
        return round(total_seconds / 60, 4)
      case "hour":
        return round(total_seconds / 3600, 4)
      case _:
        return None
  except ValueError:
    return None


def process_book_df(df: pd.DataFrame) -> pd.DataFrame:
  # Remove the tvshow
  df = df[~df["audio_url"].str.contains("tvshows", na=False)].copy()

  # Add new columns (initialized to None)
  df["sample_rate"] = None
  df["quality"] = None
  df["word_count"] = None
  df["num_sentences"] = None
  df["audio_size"] = None
  df["text_size"] = None

  return df


def convert_metadata_to_csv(metadata_dir: Path, metadata_book_path: Path):
  """
  Reads JSON metadata files, saves all metadata to a single file as CSV.
  """
  metadata_dir.mkdir(parents=True, exist_ok=True)

  json_files = list(metadata_dir.glob("*.json"))
  all_metadata = []

  logger.info(f"Processing {len(json_files)} metadata JSON files from {metadata_dir}")

  for json_file in tqdm(json_files, desc="Converting metadata to CSV"):
    with open(json_file, "r", encoding="utf-8") as f:
      try:
        data = json.load(f)

        if "duration" in data and isinstance(data["duration"], str):
          data["duration_hours"] = convert_duration(data["duration"], "hour")

        if "narrator" in data:
          if isinstance(data["narrator"], dict):
            data["narrator"] = [data["narrator"]]
          elif not isinstance(data["narrator"], list):
            data["narrator"] = [{"name": str(data["narrator"]), "url": ""}]

        all_metadata.append(data)
      except json.JSONDecodeError:
        logger.warning(f"Error parsing JSON file: {json_file}. Skipping.")
      except Exception as e:
        logger.error(f"Unexpected error processing {json_file}: {e}")

  if all_metadata:
    df = pd.DataFrame(all_metadata)

    max_narrators = 0
    for entry in all_metadata:
      if "narrator" in entry and isinstance(entry["narrator"], list):
        max_narrators = max(max_narrators, len(entry["narrator"]))

    narrator_cols = {}
    for i in range(max_narrators):
      narrator_cols[f"narrator_{i + 1}_name"] = df["narrator"].apply(
        lambda x: x[i]["name"] if isinstance(x, list) and len(x) > i else None
      )
      narrator_cols[f"narrator_{i + 1}_url"] = df["narrator"].apply(
        lambda x: x[i]["url"] if isinstance(x, list) and len(x) > i else None
      )

    narrator_df_expanded = pd.DataFrame(narrator_cols)
    df = pd.concat([df.drop(columns=["narrator"]), narrator_df_expanded], axis=1)

    df = process_book_df(df)

    df.to_csv(metadata_book_path, index=False)
    logger.info(
      f"Metadata saved to {metadata_book_path}. {len(all_metadata)} files processed."
    )
  else:
    logger.info("No metadata files were processed, so no CSV was created.")


def get_narrator_metadata(config_obj: Config) -> pd.DataFrame:
  """
  Get metadata for each narrator from google sheet file using the config object.
  """
  try:
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv"}
    response = requests.get(
      config_obj.narrator_download_url, headers=headers, allow_redirects=True
    )
    response.raise_for_status()

    if "text/html" in response.headers.get("content-type", "").lower():
      logger.error(
        "Received HTML instead of CSV from Google Sheet URL. Check the URL and sharing settings."
      )
      logger.debug(f"HTML content returned (first 500 chars): {response.text[:500]}")
      raise ValueError("Received HTML instead of CSV")

    df = pd.read_csv(
      io.StringIO(response.content.decode("utf-8")), dtype=str, keep_default_na=False
    )
    config_obj.metadata_narrator_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config_obj.metadata_narrator_path, index=False, encoding="utf-8")
    logger.info(f"Narrator metadata saved to {config_obj.metadata_narrator_path}")

    return df

  except Exception as e:
    logger.error(f"Error fetching narrator metadata: {e}")
    return pd.DataFrame()


def get_valid_audio_urls(metadata_book_path: Path) -> List[str]:
  """
  Get a list of valid audio URLs from the metadata CSV file.
  """
  if not metadata_book_path.exists():
    logger.warning(
      f"Metadata book CSV not found at {metadata_book_path}. Returning empty list."
    )
    return []
  try:
    df = pd.read_csv(metadata_book_path)
    if "audio_url" in df.columns:
      return df["audio_url"].dropna().tolist()
    else:
      logger.warning(
        f"Column 'audio_url' not found in {metadata_book_path}. Returning empty list."
      )
      return []
  except Exception as e:
    logger.error(f"Error reading metadata book CSV for audio URLs: {e}")
    return []


@app.command(name="run", help="Run the audiobook metadata pipeline.")
async def main(
  check_urls: bool = typer.Option(
    False,
    "--check-urls",
    "-c",
    help="Force to check the existence of all both text and audiobook URLs.",
  ),
  fetch_metadata: bool = typer.Option(
    False,
    "--fetch-metadata",
    "-f",
    help="Force to fetch metadata for each book and save as JSON.",
  ),
  create_book_metadata_csv: bool = typer.Option(
    False,
    "--create-book-metadata-csv",
    help="Process and convert JSON metadata files to a single CSV file for books.",
  ),
  fetch_narrator_metadata: bool = typer.Option(
    False,
    "--fetch-narrator-metadata",
    help="Fetch and save narrator metadata from Google Sheet to CSV.",
  ),
):
  """
  Main function to run various stages of the audiobook data pipeline.
  This version focuses solely on metadata processing, without direct download steps.
  """
  # Ensure base directories exist (still useful for metadata saving)
  config.audio_raw_dir.mkdir(parents=True, exist_ok=True)
  config.text_pdf_dir.mkdir(parents=True, exist_ok=True)
  config.metadata_dir.mkdir(parents=True, exist_ok=True)

  # Get all book's URLs
  if check_urls or not config.valid_book_urls_path.exists():
    logger.info("Getting all audiobook URLs and names")
    text_audio_urls: pd.DataFrame = await utils.get_all_book_url()
    invalid_count = len(text_audio_urls[text_audio_urls["source"] == "invalid"])
    total_count = len(text_audio_urls)
    logger.info(
      f"Found {invalid_count} ({invalid_count / total_count * 100:.2f}%) invalid audiobooks"
    )

    async with aiofiles.open(config.valid_book_urls_path, "w") as f:
      valid_urls = [
        f"{row['audio_url']}, {row['text_url']}, {row['source']}"
        for _, row in text_audio_urls.iterrows()
      ]
      await f.write("\n".join(valid_urls))

  # Fetch book metadata
  if fetch_metadata:
    logger.info(f"Loading all audiobook URLs from {config.valid_book_urls_path}")
    audio_urls, text_urls = await load_urls()
    await fetch_book_metadata(text_urls, audio_urls, config.metadata_dir)

  # Create book metadata CSV
  if create_book_metadata_csv:
    logger.info(
      "Processing and converting book metadata JSON files to a single CSV file"
    )
    await asyncio.to_thread(
      convert_metadata_to_csv, config.metadata_dir, config.metadata_book_path
    )

  # Fetch narrator metadata
  if fetch_narrator_metadata:
    logger.info("Fetching narrator metadata from Google Sheet")
    await asyncio.to_thread(get_narrator_metadata, config)


if __name__ == "__main__":
  app()
