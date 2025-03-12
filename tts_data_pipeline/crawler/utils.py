import asyncio
import json
import os
import string
from typing import Dict, List, Tuple

import aiofiles
import aiohttp
import httpx
from rich import print
from selectolax.parser import HTMLParser
from tqdm.asyncio import tqdm, tqdm_asyncio

from tts_data_pipeline import constants


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

    return book_urls


def remove_punctuations(sentence: str):
    translator = str.maketrans("", "", string.punctuation)
    return sentence.translate(translator)


async def fetch_download_audio_url(book_url: str) -> List[str]:
    """Fetch all download URLs for a given book."""
    parser = await get_web_content(book_url)
    return [node.attributes.get("href") for node in parser.css("a.ai-track-btn")]


# async def fetch_download_text_url(book_url: str) -> str:
#     """Fetch the download URL for the text of a given book."""
#     parser = await get_web_content(book_url)
#     return parser.css_first("a[id=download-pdf]").attributes.get("href")


# * No longer using
# async def download_audio(url: str, file_path: str):
#     """
#     Download audio file asynchronously with progress tracking.

#     Args:
#         url (str): The download URL of the audio file.
#         file_path (str): The file path to save the downloaded audio file.
#     """
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     async with httpx.AsyncClient(
#         headers={"User-Agent": constants.USER_AGENT}
#     ) as client:
#         async with client.stream("GET", url) as response:
#             response.raise_for_status()
#             total_size = int(response.headers.get("Content-Length", 0))

#             async with aiofiles.open(file_path, "wb") as file:
#                 with tqdm(
#                     total=total_size,
#                     unit="B",
#                     unit_scale=True,
#                     unit_divisor=1024,
#                     desc=f"Downloading {os.path.basename(file_path)}",
#                 ) as progress:
#                     async for chunk in response.aiter_bytes(
#                         chunk_size=8192
#                     ):  # Increased chunk size
#                         await file.write(chunk)
#                         progress.update(len(chunk))


# async def download_text(url: str, file_path: str):
#     """
#     Download a PDF file from the given URL.

#     Args:
#         url (str): The download URL of the PDF.
#         file_path (str): The file path to save the downloaded PDF.
#     """
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)

#     async with httpx.AsyncClient(
#         headers={"User-Agent": constants.USER_AGENT}
#     ) as client:
#         async with client.stream(
#             "GET", url
#         ) as response:  # Changed to stream for consistency
#             response.raise_for_status()
#             total_size = int(response.headers.get("Content-Length", 0))

#             async with aiofiles.open(file_path, "wb") as file:
#                 with tqdm(
#                     total=total_size,
#                     unit="B",
#                     unit_scale=True,
#                     unit_divisor=1024,
#                     desc=f"Downloading {os.path.basename(file_path)}",
#                 ) as progress:
#                     async for chunk in response.aiter_bytes(
#                         chunk_size=8192
#                     ):  # Fixed to use async iteration
#                         await file.write(chunk)  # Added await
#                         progress.update(len(chunk))


async def download_by_cli(url: str, file_path: str):
    """
    Download file using aria2c with progress tracking.

    Args:
        url (str): The download URL of the audio file.
        save_path (str): The file path to save the downloaded audio file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Configure aria2c command

    # aria2_cmd = [
    #     "aria2c",
    #     url,
    #     f"--dir={os.path.dirname(file_path)}",
    #     "--max-connection-per-server=16",  # Multiple connections for faster downloads
    #     "--min-split-size=1M",  # Split downloads into chunks
    #     "--console-log-level=warn",  # Reduce log output
    #     "--summary-interval=0",  # Disable summary output
    #     "--download-result=hide",  # Hide download result
    #     # f'--user-agent="{constants.USER_AGENTS}"',  # Add User-Agent header
    # ]

    cmd = [
        "wget",
        url,
        f"-P={os.path.dirname(file_path)}",
        "--quite",
        f"--user-agent={constants.USER_AGENTS}",
    ]

    # Start the process
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Get process results and check for errors
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        error_msg = stderr.decode("utf-8", errors="replace")
        print(f"Download failed with exit code {process.returncode}: {error_msg}")


async def download_full_book(
    audio_url: str,
    text_url: str,
    audio_save_path: str,
    text_save_path: str,
):
    """
    Fetch download URLs and download all audio parts of a book
    in both text source and audio source.
    """
    book_name = audio_url.split("/")[-1]

    try:
        # Fetch both URLs concurrently
        audio_download_urls, text_download_url = await asyncio.gather(
            fetch_download_audio_url(audio_url), text_url
        )

        tasks = []
        text_path = os.path.join(text_save_path, f"{book_name}.pdf")
        tasks.append(download_by_cli(text_download_url, text_path))

        for idx, url in enumerate(audio_download_urls):
            audio_path = os.path.join(audio_save_path, f"{book_name}_{idx}.mp3")
            tasks.append(download_by_cli(url, audio_path))

        await asyncio.gather(*tasks)
    except httpx.HTTPStatusError:
        return


# * No longer using
# async def check_a_book(
#     name: str, session: aiohttp.ClientSession, timeout: int = 5
# ) -> bool:
#     """Check if a book exists using aiohttp, handling timeout errors."""
#     try:
#         async with session.head(
#             url=constants.TEXT_BASE_URL + name, timeout=timeout
#         ) as response:
#             return response.status == 200
#     except (aiohttp.ClientError, asyncio.TimeoutError):
#         # Handle both client errors and timeouts
#         return False


# async def check_exists(urls: List[str], timeout: int = 5) -> List[str]:
#     """Check all audiobooks in parallel and return valid URLs with progress tracking.

#     Args:
#         urls: List of URLs to check
#         timeout: Maximum time in seconds to wait for each request before considering it failed

#     Returns:
#         List of URLs that exist (return 200 status code)
#     """

#     print("Checking the existence of the book in the text source")
#     async with aiohttp.ClientSession(
#         headers={"User-Agent": constants.USER_AGENTS}
#     ) as session:
#         # Create the tasks
#         tasks = [check_a_book(url.split("/")[-1], session, timeout) for url in urls]

#         # Use tqdm_asyncio to track progress
#         results = await tqdm_asyncio.gather(*tasks, desc="Checking URLs")

#     # Filter valid URLs
#     return [url for url, exist in zip(urls, results) if exist]


async def get_web_content(url: str) -> HTMLParser:
    """
    Asynchronously fetch HTML content from a given URL.

    Args:
        url (str): The audio URL.

    Returns:
        HTMLParser: Parsed HTML content.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
        return HTMLParser(response.text)


async def download_with_semaphore(
    audio_url, text_url, download_semaphore: asyncio.Semaphore
):
    async with download_semaphore:
        return await download_full_book(
            audio_url,
            text_url,
            constants.RAW_DIR,
            constants.PDF_DIR,
        )


async def get_metadata(
    text_url: str, audio_url: str, semaphore: asyncio.Semaphore, save_path: str = None
) -> Dict[str, str]:
    """
    Asynchronously get audio metadata from an book URL.

    Args:
        text_url (str): The URL of the book page.
        audio_url (str): The URL of the audiobook page.
        save (bool): Whether to save the metadata as a JSON file.

    Returns:
        Dict[str, str]: The book metadata, containing title, url, duration, author, and narrator's name.
    """
    # // TODO: Change the css selector for tiemsach.org, not sachnoiviet
    async with semaphore:
        try:
            text_parser = await get_web_content(text_url)
            audio_parser = await get_web_content(audio_url)
        except httpx.HTTPStatusError:
            return {}

        try:
            author = text_parser.css_first("div.book-metadata a strong").text(
                strip=True
            )
            duration = audio_parser.css_first(".featu").text(strip=True)
            narrator = audio_parser.css_first("i.fa-microphone + a").text(strip=True)
        except AttributeError:
            author, duration, narrator = "", "", ""

        metadata = {
            "audio_url": audio_url,
            "text_url": text_url,
            "title": audio_url.split("/")[-1],
            "author": author,
            "duration": duration,
            "narrator": narrator,
        }

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_path += f"{text_url.split('/')[-1]}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
        else:
            print("Don't save any book's metadata")

        return metadata
