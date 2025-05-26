import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from typing import Optional, Union

from tts_data_pipeline import Book, constants
from tts_data_pipeline.alignment.utils import (
  book_alignment,
  check_dependencies,
  logger,
)


def parse_args():
  parser = argparse.ArgumentParser(description="Audio to text alignment")
  parser.add_argument(
    "-a",
    "--audio",
    type=str,
    help="Path to the audio file to align",
    default="UNSPECIFIED",
  )
  parser.add_argument(
    "-t",
    "--text",
    type=str,
    help="Path to the text file to align",
    default="UNSPECIFIED",
  )
  parser.add_argument(
    "-j",
    "--jobs",
    type=int,
    help="Number of parallel jobs to run (-1 = use all cores) (default: 1)",
    default=1,
  )
  parser.add_argument(
    "-s",
    "--split",
    action="store_true",
    help="Split text and audio file to samples based on alignment data (default: False)",
    default=False,
  )
  parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    help="Force to align audio and text although their alignment data exists (default: False)",
    default=False,
  )

  return parser.parse_args()


def find_json(audio_path: Union[str, Path]) -> Optional[Path]:
  """
  Find the JSON metadata file for an audio file.

  Args:
      audio_path: Path to the audio file

  Returns:
      Path to the JSON file if it exists, None otherwise
  """
  # Convert to Path object for consistent handling
  audio_path = Path(audio_path)

  # Create the expected JSON path
  json_name = audio_path.name.replace(".wav", ".json")
  json_path = Path(constants.METADATA_SAVE_PATH) / json_name

  if json_path.exists():
    return json_path
  else:
    logger.error(f"Metadata file not found: {json_path}")
    return None


def validate_files(audio_path: Path, text_path: Path) -> bool:
  """
  Validate that the audio and text files exist and have the correct extensions.

  Args:
      audio_path: Path to the audio file
      text_path: Path to the text file

  Returns:
      True if files are valid, False otherwise
  """
  # Check if files exist
  if not audio_path.exists():
    logger.error(f"Audio file not found: {audio_path}")
    return False

  if not text_path.exists():
    logger.error(f"Text file not found: {text_path}")
    return False

  # Check file extensions
  if audio_path.suffix.lower() not in [".wav", ".mp3", ".flac"]:
    logger.error(f"Unsupported audio format: {audio_path.suffix}")
    return False

  if text_path.suffix.lower() not in [".txt", ".md"]:
    logger.error(f"Unsupported text format: {text_path.suffix}")
    return False

  return True


def find_alignment_output(book: Book):
  alignment_path = Path(constants.AENEAS_OUTPUT_DIR) / book.name / "output.tsv"

  if alignment_path.exists():
    return alignment_path
  else:
    logger.error(f"Alignment file not found: {alignment_path}")
    return None


def main():
  # Check dependencies
  if not check_dependencies():
    return

  # Get arguments
  args = parse_args()
  audio_path = Path(args.audio)
  text_path = Path(args.text)
  if not validate_files(audio_path, text_path):
    return

  # Align audio and text
  json_path = find_json(audio_path)
  if json_path:
    book = Book.from_json(json_path)
    book.update_paths(text_path=text_path, audio_path=audio_path)
  else:
    logger.error(f"Not able to setup book. {audio_path=}, {text_path=}")
    return

  alignment_path = find_alignment_output(book)
  if alignment_path and not args.force:
    book.update_paths(alignment_path=alignment_path)

  # Run alignment
  jobs = os.cpu_count() if args.jobs == -1 else args.jobs
  try:
    with ThreadPoolExecutor(max_workers=jobs) as executor:
      future = executor.submit(
        book_alignment, book, args.split, remove_first=True, jobs=jobs
      )
      future.result()
  except Exception as e:
    logger.exception(f"Error alignment {audio_path=}, {text_path=}: {e}")


if __name__ == "__main__":
  main()
