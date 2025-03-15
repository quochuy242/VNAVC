import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List

import httpx
import pandas as pd

from tts_data_pipeline import constants

from . import utils


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

        author = text_parser.css_first(
            "div.product-price span.text-brand"
        )  # The text source is more reliable than audio one
        duration = audio_parser.css_first(".featu")
        narrator = audio_parser.css_first("i.fa-microphone + a")

        metadata = {
            "audio_url": audio_url,
            "text_url": text_url,
            "title": audio_url.split("/")[-1],
            "author": author.text(strip=True) if author else "",
            "duration": duration.text(strip=True) if duration else "",
            "narrator": narrator.text(strip=True) if narrator else "",
        }

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_path += f"{text_url.split('/')[-1]}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
        else:
            print("Don't save any book's metadata")

        return metadata


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

    def process_df(df: pd.DataFrame) -> pd.DataFrame:
        # Convert duration to hours
        df["duration_hour"] = df["duration"].apply(convert_duration, unit="hour")
        # Remove the tvshow
        df = df[~df["audio_url"].str.contains("tvshows", na=False)]
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
