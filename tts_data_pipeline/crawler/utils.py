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
from rich.progress import (
  Progress,
  SpinnerColumn,
  BarColumn,
  TextColumn,
  TimeElapsedColumn,
)
from rich.console import Console
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


def query_download_url(
  query: Optional[str],
  name: Optional[str] = None,
  author: Optional[str] = None,
  narrator: Optional[str] = None,
  random: int = 0,
) -> Tuple[List[str], List[str]]:
  """
  Get a list of valid audio URLs from the metadata CSV file.
  """
  try:
    df = pd.read_csv(constants.METADATA_BOOK_PATH)
  except FileNotFoundError:
    logger.error(
      f"Metadata file not found at {constants.METADATA_BOOK_PATH}. Please run the metadata fetching argument first."
    )
    return ([], [])

  if random > 0:
    return (
      randomlib.sample(df["audio_download_url"].tolist(), random),
      randomlib.sample(df["text_download_url"].tolist(), random),
    )

  if query == "all":
    return df["audio_download_url"].tolist(), df["text_download_url"].tolist()
  else:
    mask = pd.Series([True] * len(df))
    if name:
      mask &= df["name"].str.contains(name, na=False)
    if author:
      mask &= df["author"].str.contains(author, na=False)
    if narrator:
      mask &= df["narrator"].str.contains(narrator, na=False)
    return (
      df[mask]["audio_download_url"].tolist(),
      df[mask]["text_download_url"].tolist(),
    )


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


async def get_text_download_url(
  name: str, source: str = "thuviensach", console: Console = None
) -> str:
  """
  Args:
    name (str): The name of the book.
    source (str): The source name of the book, example "thuviensach", "taisachhay".

  Returns:
      str: The text book URL.
  """
  if source == "thuviensach":
    return f"{constants.TEXT_DOWNLOAD_URL[source]}{name}.pdf"
  elif source == "taisachhay":
    content = await get_web_content(f"{constants.TEXT_DOWNLOAD_URL[source]}{name}/")
    if not content:
      console.log(
        f"Failed to fetch content for [blue]{name}[/blue] from [yellow]{source}[/yellow]"
      ) if console else None
      return ""
    pdf_link = ""
    for tr in content.css("tr"):
      if "PDF" in tr.text():
        a_tag = tr.css_first("a")
        if a_tag:
          pdf_link = a_tag.attributes.get("href")
          break
    console.log(
      f"PDF link found: {pdf_link if pdf_link else 'None'}"
    ) if console else None
    return pdf_link if pdf_link else ""
  else:
    return ""  # Unsupported source


async def get_text_book_url(name: str, source: str = "thuviensach") -> str:
  """
  Get the text book URL from the source.

  Args:
      name (str): The name of the book.
      source (str): The source of the book, example "thuviensach", "taisachhay".

  Returns:
      str: The text book URL.
  """
  return (
    constants.TEXT_BASE_URL[source] + name
    if source != "invalid"
    else constants.TEXT_BASE_URL["thuviensach"] + name  # Default to thuviensach
  )


async def get_web_content(url: str, console: Console = None) -> Optional[HTMLParser]:
  """
  Asynchronously fetch HTML content from a given URL.

  Args:
      url (str): The audio URL.

  Returns:
      HTMLParser: Parsed HTML content.
  """
  async with httpx.AsyncClient(
    timeout=30, headers={"User-Agent": randomlib.choice(constants.USER_AGENTS)}
  ) as client:
    response = await client.get(url)
    return (
      HTMLParser(response.text)
      if response.status_code < 400
      else console.log(
        f"Failed to fetch HTML content for {url}: {response.status_code}"
      )
      if console
      else None
    )


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


async def print_status(console: Console, url: str, status_code: int):
  """
  Print the status of the URL check without breaking the progress bar.

  Args:
      console (Console): The rich Console instance used by Progress.
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
  console.log(
    f"Checking URL: {url} - Status: [bold {color}]{status_code}[/bold {color}]"
  )


async def double_check_url(
  url: str, console: Console, semaphore: asyncio.Semaphore
) -> str:
  """
  Double check if the URL is valid in the text source.
  """
  try:
    for source in constants.TEXT_DOWNLOAD_URL.keys():
      text_download_url = await get_text_download_url(url.split("/")[-1], source)
      if not text_download_url:
        continue

      async with semaphore:
        async with httpx.AsyncClient(
          timeout=30, headers={"User-Agent": randomlib.choice(constants.USER_AGENTS)}
        ) as client:
          response = await client.head(text_download_url)
          status_code = response.status_code
          await print_status(console, text_download_url, status_code)

          if status_code < 400:
            return source

    return "invalid"

  except httpx.RequestError as e:
    logger.exception(f"Request error for {text_download_url}: {e}")
    return "invalid"


async def get_all_book_url() -> pd.DataFrame:
  # Get all audiobook URLs from the categories on the website.
  categories = [
    "kinh-te-khoi-nghiep",
    "tam-linh-ton-giao",
    "truyen-tieu-thuyet",
    "tu-duy-ky-nang",
    "tu-lieu-lich-su",
  ]

  category_urls = [f"{constants.AUDIO_CATEGORY_URL}{cat}" for cat in categories]
  num_pages = await asyncio.gather(*(get_num_page(url) for url in category_urls))

  page_urls = []
  for url, pages in zip(category_urls, num_pages):
    page_urls.append(url)
    page_urls.extend([f"{url}/page/{p}" for p in range(2, pages + 1)])

  fetch_semaphore = asyncio.Semaphore(constants.FETCH_URL_LIMIT)

  async def get_web_content_limited(url: str) -> HTMLParser:
    async with fetch_semaphore:
      return await get_web_content(url)

  parsers = await asyncio.gather(*(get_web_content_limited(url) for url in page_urls))

  # Extract audio URLs from the parsers
  # Each parser corresponds to a page, and we extract audio URLs from each page
  # The audio URLs are in the format of <div class="poster"><a href="...">
  # We use a list comprehension to flatten the list of lists
  # and get all audio URLs in a single list
  audio_urls = [
    node.attributes.get("href")
    for parser in parsers
    for node in parser.css("div.poster a")
  ]
  audio_urls = [url for url in audio_urls if url is not None]

  # Check if audio URLs are valid in text sources
  if not audio_urls:
    logger.error("No audio URLs found. Please check the website structure.")
    return pd.DataFrame(columns=["audio_url", "text_url", "source"])
  console = Console()
  audio_urls_with_text_source = defaultdict(list)
  check_semaphore = asyncio.Semaphore(constants.CHECK_URL_LIMIT)
  with Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
    console=console,
  ) as progress:
    task = progress.add_task("Checking URLs...", total=len(audio_urls))

    async def process_url(url: str):
      source = await double_check_url(url, console, check_semaphore)
      asyncio.sleep(0.1)  # Small delay to avoid overwhelming the server
      audio_urls_with_text_source[source].append(url)
      progress.update(task, advance=1)

    await asyncio.gather(*(process_url(url) for url in audio_urls))

  # Create a DataFrame with audio URLs and their corresponding text URLs
  text_audio_urls = []
  for source, urls in audio_urls_with_text_source.items():
    for audio_url in urls:
      text_url = await get_text_book_url(audio_url.split("/")[-1], source)
      text_audio_urls.append((audio_url, text_url, source))

  return pd.DataFrame(text_audio_urls, columns=["audio_url", "text_url", "source"])


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
