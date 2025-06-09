import asyncio
import os
import os.path as osp
import random as randomlib
import shutil
from collections import defaultdict
from typing import List, Optional, Tuple

import httpx
import pandas as pd
from loguru import logger
from playwright.async_api import async_playwright
from rich import print
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


async def get_text_download_url(name: str, source: str = "thuviensach") -> str:
  if source == "thuviensach":
    return f"{constants.TEXT_DOWNLOAD_URL[source]}{name}.pdf"
  elif source == "taisachhay":
    content = await get_web_content(f"{constants.TEXT_DOWNLOAD_URL[source]}{name}/")
    for tr in content.css("tr"):
      if ".PDF" in tr.text():
        a_tag = tr.css_first("a")
        if a_tag:
          pdf_link = a_tag.attributes.get("href")
          break
    return pdf_link if pdf_link else ""


async def get_web_content(url: str) -> HTMLParser:
  """
  Asynchronously fetch HTML content from a given URL.

  Args:
      url (str): The audio URL.

  Returns:
      HTMLParser: Parsed HTML content.
  """
  print(f"Fetching content from {url}")
  async with httpx.AsyncClient(
    timeout=30, headers={"User-Agent": randomlib.choice(constants.USER_AGENTS)}
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


async def get_all_audiobook_url() -> Tuple[List[str]]:
  """
  Asynchronously fetch all audiobook URLs from different categories.

  Returns:
      List[str]: A list of all audiobook URLs
  """

  async def print_status(url: str, status_code: int):
    """
    Print the status of the URL check.

    Args:
        url (str): The URL being checked.
        status_code (int): The HTTP status code returned.
    """
    if status_code < 300:
      color = "green"
    elif status_code < 400:
      color = "cyan"
    elif status_code < 500:
      color = "yellow"
    else:
      color = "red"
    print(f"Checking URL: {url} - Status: [bold {color}]{status_code}[/bold {color}]")

  async def double_check_url(url: str) -> str:
    """
    Double check if the URL is valid in the text source.

    Args:
        url (str): The audio URL to check.

    Returns:
        bool: The result of the check, True if valid, False otherwise.
    """
    text_download_url = await get_text_download_url(url.split("/")[-1])
    alternate_text_download_url = (
      constants.TEXT_DOWNLOAD_URL[-1] + url.split("/")[-1] + "/"
    )
    try:
      async with httpx.AsyncClient(
        timeout=30, headers={"User-Agent": randomlib.choice(constants.USER_AGENTS)}
      ) as client:
        response = await client.head(text_download_url)
        status_code = response.status_code
        print_status(text_download_url, status_code)

        if response.status_code < 400:
          return "thuviensach"
        # If the first URL is not valid, check the alternate URL
        else:
          response = await client.head(alternate_text_download_url)
          status_code = response.status_code
          print_status(alternate_text_download_url, status_code)
          return "taisachhay" if response.status_code < 400 else "invalid"

    except httpx.RequestError as e:
      logger.error(f"Request error for {text_download_url}: {e}")
      return False

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

  semaphore = asyncio.Semaphore(constants.FETCH_URL_LIMIT)  # Limit concurrent requests

  async def get_web_content_limited(url: str) -> HTMLParser:
    async with semaphore:
      return await get_web_content(url)

  parsers = await asyncio.gather(*(get_web_content_limited(url) for url in page_urls))

  # Extract all audiobook URLs from each page
  book_urls = [
    node.attributes.get("href")
    for parser in parsers
    for node in parser.css("div.poster a")
  ]

  # Remove None values
  book_urls = [url for url in book_urls if url is not None]

  # Double check if the URL is valid in the text source
  unvalid_urls = []
  main_valid_urls = []
  alternate_valid_urls = []
  for url in book_urls:
    result = await double_check_url(url)
    if result == "invalid":
      unvalid_urls.append(url)
    elif result == "thuviensach":
      main_valid_urls.append(url)
    elif result == "taisachhay":
      alternate_valid_urls.append(url)

  return main_valid_urls, alternate_valid_urls, unvalid_urls


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
