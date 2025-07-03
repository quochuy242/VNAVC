import ast
import asyncio
import os
import random as randomlib
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Optional

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm

from tts_data_pipeline import constants
from tts_data_pipeline.crawler.utils import logger


class OptimizedDownloader:
  def __init__(
    self,
    max_concurrent_books: int = 3,
    max_concurrent_files_per_book: int = 8,
    max_retries: int = 3,
    chunk_size: int = 8192,  # 8KB chunks for better memory usage
    timeout: int = 30,
  ):
    self.max_concurrent_books = max_concurrent_books
    self.max_concurrent_files_per_book = max_concurrent_files_per_book
    self.max_retries = max_retries
    self.chunk_size = chunk_size
    self.timeout = timeout

    # Create semaphores for rate limiting
    self.book_semaphore = asyncio.Semaphore(max_concurrent_books)
    self.session = None

  async def __aenter__(self):
    # Configure aiohttp session with optimizations
    connector = aiohttp.TCPConnector(
      limit=100,  # Total connection pool size
      limit_per_host=20,  # Max connections per host
      ttl_dns_cache=300,  # DNS cache TTL
      use_dns_cache=True,
      keepalive_timeout=30,
      enable_cleanup_closed=True,
    )

    timeout = aiohttp.ClientTimeout(total=self.timeout)

    self.session = aiohttp.ClientSession(
      connector=connector,
      timeout=timeout,
      headers={"User-Agent": randomlib.choice(constants.USER_AGENTS)},
    )
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    if self.session:
      await self.session.close()

  async def download_file(
    self, url: str, save_path: Path, description: str = "", retry_count: int = 0
  ) -> bool:
    """Download a single file with retry logic"""
    try:
      save_path.parent.mkdir(parents=True, exist_ok=True)

      # Rotate User-Agent for each request
      headers = {"User-Agent": randomlib.choice(constants.USER_AGENTS)}

      async with self.session.get(
        url, headers=headers, timeout=self.timeout
      ) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))

        # Use async progress bar
        progress_bar = tqdm(
          total=total_size,
          unit="B",
          unit_scale=True,
          unit_divisor=1024,
          desc=description or save_path.name,
          ncols=100,
          leave=False,
        )

        try:
          with open(save_path, "wb") as f:
            async for chunk in response.content.iter_chunked(self.chunk_size):
              if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
        finally:
          progress_bar.close()

        logger.info(f"Successfully downloaded: {save_path.name}")
        return True

    except asyncio.TimeoutError:
      logger.warning(f"Timeout downloading {url} (attempt {retry_count + 1})")
    except aiohttp.ClientError as e:
      logger.warning(f"Client error downloading {url}: {e} (attempt {retry_count + 1})")
    except Exception as e:
      logger.error(
        f"Unexpected error downloading {url}: {e} (attempt {retry_count + 1})"
      )

    # Retry logic
    if retry_count < self.max_retries:
      await asyncio.sleep(2**retry_count)  # Exponential backoff
      return await self.download_file(url, save_path, description, retry_count + 1)

    logger.error(f"Failed to download {url} after {self.max_retries + 1} attempts")
    return False

  async def download_book_files(
    self,
    audio_urls: List[str],
    text_url: str,
    book_name: str,
    audio_save_path: Path,
    text_save_path: Path,
  ) -> bool:
    """Download all files for a single book with controlled parallelism"""

    # Prepare download tasks
    download_tasks = []

    # Text file download
    text_filename = f"{book_name}.pdf"
    text_full_path = text_save_path / text_filename
    download_tasks.append(
      self.download_file(text_url, text_full_path, f"Text: {book_name}")
    )

    # Audio files download
    book_audio_dir = audio_save_path / book_name
    for idx, audio_url in enumerate(audio_urls, start=1):
      # Extract extension from URL
      ext = Path(audio_url.split("/")[-1]).suffix or ".mp3"
      audio_filename = f"{book_name}_{idx}{ext}"
      audio_full_path = book_audio_dir / audio_filename

      download_tasks.append(
        self.download_file(
          audio_url, audio_full_path, f"Audio {idx}/{len(audio_urls)}: {book_name}"
        )
      )

    # Execute downloads with controlled concurrency per book
    file_semaphore = asyncio.Semaphore(self.max_concurrent_files_per_book)

    async def download_with_semaphore(task):
      async with file_semaphore:
        return await task

    try:
      # Run all downloads for this book
      results = await asyncio.gather(
        *[download_with_semaphore(task) for task in download_tasks],
        return_exceptions=True,
      )

      # Check results
      success_count = sum(1 for r in results if r is True)
      total_count = len(results)

      if success_count == total_count:
        logger.info(f"Successfully downloaded all files for: {book_name}")
        return True
      else:
        logger.warning(
          f"Downloaded {success_count}/{total_count} files for: {book_name}"
        )
        return False

    except Exception as e:
      logger.error(f"Error downloading book {book_name}: {e}")
      # Cleanup on failure
      if book_audio_dir.exists():
        shutil.rmtree(book_audio_dir, ignore_errors=True)
      if text_full_path.exists():
        text_full_path.unlink(missing_ok=True)
      return False

  async def download_book_with_semaphore(
    self,
    audio_urls: List[str],
    text_url: str,
    book_name: str,
    audio_save_path: Path,
    text_save_path: Path,
  ) -> bool:
    """Download a book with global semaphore limiting"""
    async with self.book_semaphore:
      return await self.download_book_files(
        audio_urls, text_url, book_name, audio_save_path, text_save_path
      )

  async def download_all_books(
    self,
    metadata_df: pd.DataFrame,
    audio_save_path: str = "./data/audio/raw/",
    text_save_path: str = "./data/text/pdf/",
  ) -> Tuple[int, int]:
    """Download all books from metadata"""

    audio_path = Path(audio_save_path)
    text_path = Path(text_save_path)

    # Create base directories
    audio_path.mkdir(parents=True, exist_ok=True)
    text_path.mkdir(parents=True, exist_ok=True)

    tasks = []

    for _, row in metadata_df.iterrows():
      book_name = row["name"]
      text_url = row["text_download_url"]

      try:
        audio_urls = ast.literal_eval(row["audio_download_url"])
        if not isinstance(audio_urls, list):
          logger.warning(f"Invalid audio URLs for {book_name}: {audio_urls}")
          continue
      except (ValueError, SyntaxError) as e:
        logger.error(f"Failed to parse audio URLs for {book_name}: {e}")
        continue

      tasks.append(
        self.download_book_with_semaphore(
          audio_urls=audio_urls,
          text_url=text_url,
          book_name=book_name,
          audio_save_path=audio_path,
          text_save_path=text_path,
        )
      )

    # Execute all book downloads
    logger.info(f"Starting download of {len(tasks)} books...")
    start_time = time.time()

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successes and failures
    successful = sum(1 for r in results if r is True)
    failed = len(results) - successful

    elapsed_time = time.time() - start_time
    logger.info(
      f"Download completed in {elapsed_time:.2f}s: "
      f"{successful} successful, {failed} failed"
    )

    return successful, failed


async def main():
  """Main function to run the optimized downloader"""
  metadata_df = pd.read_csv("./data/metadata/final_metadata.csv")

  # Configuration for optimal performance
  downloader_config = {
    "max_concurrent_books": 3,  # Adjust based on bandwidth and server limits
    "max_concurrent_files_per_book": 8,  # Files per book in parallel
    "max_retries": 3,
    "chunk_size": 8192,  # 8KB chunks for better memory usage
    "timeout": 10000,
  }

  async with OptimizedDownloader(**downloader_config) as downloader:
    successful, failed = await downloader.download_all_books(
      metadata_df,
      audio_save_path="./data/audio/raw/",
      text_save_path="./data/text/pdf/",
    )

    # Print final results
    print("\n" + "=" * 30)
    print("\n📊 Final Results:")
    print("-" * 30)
    print(f"📚 Total books processed: {len(metadata_df)}")
    print(f"✅ Successful downloads: {successful}")
    print(f"❌ Failed downloads: {failed}")
    print(f"📈 Success rate: {successful / (successful + failed) * 100:.1f}%")
    print("-" * 30)


if __name__ == "__main__":
  asyncio.run(main())
