import concurrent.futures
import glob
import os

from tqdm import tqdm

from tts_data_pipeline import constants
from tts_data_pipeline.pre_processing.audio import audio_processing, group_audiobook
from tts_data_pipeline.pre_processing.text import text_processing
from tts_data_pipeline.pre_processing.utils import logger


def main():
  # Text pre-processing
  """
  Process all PDF files in a directory:
  1. Convert them to text
  2. Split into sentences
  3. Normalize text
  4. Save to output file
  """

  logger.info("Starting PDF text processing...")

  # Get all PDF files in the directory
  pdf_files = glob.glob(os.path.join(constants.TEXT_PDF_DIR, "*.pdf"))

  # Check if any PDF files were found
  if not pdf_files:
    logger.warning(f"No PDF files found in {constants.TEXT_PDF_DIR}")
    return

  with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    text_results = list(
      tqdm(
        executor.map(text_processing, pdf_files),
        total=len(pdf_files),
        desc="Processing PDFs",
        leave=True,
        dynamic_ncols=True,
      )
    )

  successful_books = sum(1 for r in text_results if r)
  logger.success(
    f"Text pre-processing completed. {successful_books}/{len(pdf_files)} books processed successfully."
  )

  # Audio pre-processing
  """
  Process all MP3 files in a directory:
  1. Group all parts of an audiobook together
  2. Convert them to WAV format
  3. Check their sample rate
  4. Move those with sample rates below the threshold to unqualified folder
  5. Move those with sample rates above the threshold to qualified folder
  6. Combine all qualified MP3 files into a single WAV file
  """

  # Create output directories if they don't exist
  os.makedirs(constants.AUDIO_QUALIFIED_DIR, exist_ok=True)
  os.makedirs(constants.AUDIO_UNQUALIFIED_DIR, exist_ok=True)

  # Get all MP3 file paths in the audio directory
  audiobooks = group_audiobook(
    constants.AUDIO_RAW_DIR, unqualified_dir=constants.AUDIO_UNQUALIFIED_DIR
  )

  # Check no MP3 files
  if not audiobooks:
    logger.warning(f"No MP3 files found in {constants.AUDIO_RAW_DIR}")

  with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    audio_results = list(
      tqdm(
        executor.map(audio_processing, audiobooks),
        total=len(audiobooks),
        desc="Processing MP3s",
        leave=True,
        dynamic_ncols=True,
      )
    )

  successful_books = sum(1 for r in audio_results if r)
  logger.success(
    f"Audio pre-processing completed. {successful_books}/{len(audiobooks)} books processed successfully."
  )


if __name__ == "__main__":
  main()
