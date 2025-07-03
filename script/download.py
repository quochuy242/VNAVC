import ast
import os
import random as randomlib
import shutil
from typing import List

import pandas as pd
import requests
from tqdm import tqdm

from tts_data_pipeline import constants
from tts_data_pipeline.crawler.utils import logger

# Load the metadata CSV file
metadata_df = pd.read_csv("./data/metadata/final_metadata.csv")

names = metadata_df["name"].tolist()
text_urls = metadata_df["text_download_url"].tolist()
audio_urls = metadata_df["audio_download_url"].tolist()


def download(
  url: str,
  directory: str,
  filename: str = "",
  check_url: bool = True,
  chunk_size: int = 1024,
) -> str | bool:
  """
  Download a file with progress bar and save it to the specified directory.

  Args:
      url (str): File URL.
      directory (str): Output directory to save the file.
      filename (str): Optional. If not provided, filename will be taken from URL.
      check_url (bool): Whether to check the URL via HEAD request first.
      chunk_size (int): Size of chunk to download (default 1024B).

  Returns:
      str: Path to saved file on success.
      False: If download failed.
  """

  os.makedirs(directory, exist_ok=True)

  # Determine save path
  if filename:
    ext = os.path.splitext(url.split("/")[-1])[1]
    save_path = os.path.join(directory, filename + ext)
  else:
    save_path = os.path.join(directory, url.split("/")[-1])

  # Check if URL is accessible
  if check_url:
    try:
      headers = {"User-Agent": randomlib.choice(constants.USER_AGENTS)}
      response = requests.head(url, headers=headers, timeout=10)
      if response.status_code >= 400:
        logger.error(f"URL returned status {response.status_code}: {url}")
        return False
    except Exception as e:
      logger.exception(f"Failed to connect to {url}: {e}")
      return False

  try:
    headers = {"User-Agent": randomlib.choice(constants.USER_AGENTS)}
    with requests.get(url, headers=headers, stream=True, timeout=20) as r:
      r.raise_for_status()
      total_size = int(r.headers.get("Content-Length", 0))

      with (
        open(save_path, "wb") as f,
        tqdm(
          total=total_size,
          unit="B",
          unit_scale=True,
          unit_divisor=1024,
          desc=os.path.basename(save_path),
          ncols=80,
        ) as pbar,
      ):
        for chunk in r.iter_content(chunk_size=chunk_size):
          if chunk:
            f.write(chunk)
            pbar.update(len(chunk))

  except Exception as e:
    logger.exception(f"Download failed: {url}, Error: {e}")
    return False

  return save_path


def download_full_book(
  audio_url: List[str],
  text_url: str,
  audio_save_path: str,
  text_save_path: str,
):
  name_book = os.path.splitext(text_url.split("/")[-1])[0]
  logger.info(f"Downloading {name_book}")
  try:
    tasks = []
    tasks.append(
      download(text_url, text_save_path, filename=name_book, check_url=False)
    )

    for idx, url in enumerate(audio_url, start=1):
      task = download(
        url,
        os.path.join(audio_save_path, name_book),
        filename=f"{name_book}_{idx}",
        check_url=False,
      )
      tasks.append(task)

    for result in tasks:
      if result is False:
        raise Exception("Download failed")

  except KeyboardInterrupt:
    logger.error(f"Download cancelled for {name_book}.")
    if os.path.exists(os.path.join(audio_save_path, name_book)):
      shutil.rmtree(os.path.join(audio_save_path, name_book))
    text_file = os.path.join(text_save_path, f"{name_book}.txt")
    if os.path.exists(text_file):
      os.remove(text_file)
    return False

  except Exception as e:
    logger.exception(f"Unhandled exception while downloading {name_book}: {e}")
    return False

  return True


for name, text_url, audio_url in tqdm(
  zip(names, text_urls, audio_urls), desc="Downloading", total=len(names), unit="book"
):
  audio_download_urls = ast.literal_eval(audio_url)
  download_full_book(
    audio_url=audio_download_urls,
    text_url=text_url,
    audio_save_path="../data/audio/raw/",
    text_save_path="../data/text/pdf/",
  )
