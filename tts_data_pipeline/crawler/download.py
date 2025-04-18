import os
import asyncio
import httpx
from tts_data_pipeline import constants
from . import utils
from .utils import logger


async def download_by_cli(url: str, directory: str, filename: str = None):
  """
  Download file using wget

  Args:
      url (str): The download URL of the audio file.
      directory (str): The file path to save the downloaded audio file.
  """
  os.makedirs(directory, exist_ok=True)

  if filename:
    ext = os.path.splitext(url.split("/")[-1])[1]
    save_path = os.path.join(directory, filename + ext)
  else:
    save_path = os.path.join(directory, url.split("/")[-1])

  # Configure wget command
  cmd = f'wget {url} -q --user-agent "{constants.USER_AGENTS}" -O {save_path}'

  # Start the process
  process = await asyncio.create_subprocess_shell(
    cmd,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
  )

  stdout, stderr = await process.communicate()
  return


async def download_full_book(
  audio_url: str,
  text_url: str,
  audio_save_path: str,
  text_save_path: str,
):
  """
  Fetch download URLs and download a book in both text source and audio sources

  Args:
      audio_url (str): The download URL of the audio file.
      text_url (str): The download URL of the text file.
      audio_save_path (str): The file path to save the downloaded audio file.
      text_save_path (str): The file path to save the downloaded text file.
  """
  name_book = text_url.split("/")[-1].split(".")[0]
  logger.info(f"Downloading {name_book}")
  try:
    # Each downloading URL of audio is the part of the book. Contrast, the text one is a book
    audio_download_urls = await utils.fetch_download_audio_url(audio_url)

    # Download audio
    tasks = [
      download_by_cli(url, audio_save_path, filename=f"{name_book}_{idx}")
      for idx, url in enumerate(audio_download_urls, start=1)
    ]

    # Download text
    tasks.append(download_by_cli(text_url, text_save_path, filename=name_book))

    await asyncio.gather(*tasks)
  except httpx.HTTPStatusError:
    return


async def download_with_semaphore(
  audio_url: str,
  text_url: str,
  audio_save_path: str,
  text_save_path: str,
  download_semaphore: asyncio.Semaphore,
):
  async with download_semaphore:
    return await download_full_book(
      audio_url,
      text_url,
      audio_save_path,
      text_save_path,
    )
