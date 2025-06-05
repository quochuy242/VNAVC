import asyncio
import os
import os.path as osp
import random as randomlib
import shutil
from collections import defaultdict
from typing import List, Optional

import httpx
import pandas as pd
from loguru import logger
from playwright.async_api import async_playwright
from selectolax.parser import HTMLParser

from tts_data_pipeline import constants
from tts_data_pipeline.crawler.playwright_server import ensure_playwright_server_running

logger.remove()
logger.add(
  f"{constants.LOG_DIR}/crawler.log",
  level="INFO",
  rotation="10 MB",
  encoding="utf-8",
  colorize=False,
  diagnose=True,
  enqueue=True,
  format=constants.FORMAT_LOG,
)


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
      mask &= df["name"].str.contains(name, na=False)
    if author:
      mask &= df["author"].str.contains(author, na=False)
    if narrator:
      mask &= df["narrator"].str.contains(narrator, na=False)
    return df[mask]["audio_url"].tolist()


def group_audiobook(
  mp3_dir: str, unqualified_dir: str, return_group: bool = False
) -> List[List[str]]:
  """Efficiently group all parts of audiobooks based on file name prefix.

  Args:
      mp3_dir (str): Path to directory containing mp3 files.
      unqualified_dir (str): Directory to move unqualified files

  Returns:
      List[List[str]]: List of lists, where each sublist contains mp3 file paths of the same audiobook.
  """
  groups = defaultdict(list)

  for mp3_file in os.listdir(mp3_dir):
    # If the file is a directory, skip it
    if osp.isdir(osp.join(mp3_dir, mp3_file)):
      continue

    file_path = osp.join(mp3_dir, mp3_file)

    # If the file is not an MP3 file, move it to unqualified folder
    if not mp3_file.endswith(".mp3"):
      logger.warning(
        f"File {mp3_file} is not an MP3 file, move it to {unqualified_dir}"
      )
      shutil.move(file_path, unqualified_dir)
      continue

    book_name = mp3_file.split("_")[0]
    output_dir = osp.join(mp3_dir, book_name)
    os.makedirs(output_dir, exist_ok=True)
    shutil.move(file_path, output_dir)

    if return_group:
      groups[book_name].append(osp.join(output_dir, mp3_file))

  return list(groups.values()) if return_group else []


async def get_text_download_url(name: str) -> str:
  return f"{constants.TEXT_DOWNLOAD_URL}{name}.pdf"


async def get_web_content(url: str) -> HTMLParser:
  """
  Asynchronously fetch HTML content from a given URL.

  Args:
      url (str): The audio URL.

  Returns:
      HTMLParser: Parsed HTML content.
  """
  async with httpx.AsyncClient(
    timeout=30, headers={"User-Agent": constants.USER_AGENTS}
  ) as client:
    response = await client.get(url)
    response.raise_for_status()
    return HTMLParser(response.text)


async def get_num_page(url: str) -> int:
  """
  Get the number of pages from a given page.

  Args:
      url (str): The URL of the page

  Returns:
      int: The number of pages in each category
  """
  parser = await get_web_content(url)
  string = parser.css_first(
    "div.pagination span"
  ).text()  # The expect output is "Trang 1 trong X"
  num_page = int(string.split(" ")[-1])  # Get X
  return num_page


async def get_all_audiobook_url() -> List[str]:
  """
  Asynchronously fetch all audiobook URLs from different categories.

  Returns:
      List[str]: A list of all audiobook URLs
  """
  categories = [
    "kinh-te-khoi-nghiep",
    "tam-linh-ton-giao",
    "truyen-tieu-thuyet",
    "tu-duy-ky-nang",
    "tu-lieu-lich-su",
  ]

  category_urls = [
    f"{constants.AUDIO_CATEGORY_URL}{category}" for category in categories
  ]

  # Get the number of page for each category
  num_pages = await asyncio.gather(*(get_num_page(url) for url in category_urls))

  # Get the web content from each category in each page
  page_urls = []
  for url, num_page in zip(category_urls, num_pages):
    page_urls.append(url)
    page_urls.extend([f"{url}/page/{page}" for page in range(2, num_page + 1)])
  parsers = await asyncio.gather(*(get_web_content(url) for url in page_urls))

  # Extract all audiobook URLs from each page
  book_urls = [
    node.attributes.get("href")
    for parser in parsers
    for node in parser.css("div.poster a")
  ]

  # Remove None values
  book_urls = [url for url in book_urls if url is not None]
  return book_urls


async def fetch_download_audio_url(book_url: str) -> List[str]:
  """Fetch all download URLs for a given book using Playwright."""
  await ensure_playwright_server_running()  # Ensure Playwright server is running
  async with async_playwright() as p:
    browser = await p.chromium.connect("ws://0.0.0.0:3000/")
    page = await browser.new_page()
    await page.goto(book_url)

    mp3_links = await page.locator("a.ai-track-btn").evaluate_all(
      "elements => elements.map(el => el.href)"
    )

    await browser.close()
    return mp3_links
