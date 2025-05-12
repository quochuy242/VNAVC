import concurrent.futures
import os
import os.path as osp

from tqdm import tqdm

from tts_data_pipeline import constants

from tts_data_pipeline.alignment.utils import (
  logger,
  check_dependencies,
  book_alignment,
  BookInfo,
)


def main():
  """
  main -> check_dependencies -> book_alignment -> process_alignment_output -> split_audio -> split_text
  """
  logger.info("Starting alignment...")

  # Check dependencies first
  if not check_dependencies():
    logger.error("Missing dependencies. Exiting.")
    return

  # Get all audio and text directories
  audio_files = sorted(os.listdir(constants.AUDIO_QUALIFIED_DIR))
  text_files = sorted(os.listdir(constants.TEXT_SENTENCE_DIR))

  # Check if the number of audio and text files match
  if audio_files != text_files:
    logger.error("Audio and text directories do not match.")
    return

  # Prepare book processing tasks
  books = []
  for file in audio_files:
    book_name = file.split("_")[0]
    audio_path = osp.join(constants.AUDIO_QUALIFIED_DIR, file)
    text_path = osp.join(constants.TEXT_SENTENCE_DIR, file)
    books.append(
      BookInfo(
        book_name=book_name,
        audio_path=audio_path,
        text_path=text_path,
        book_id=None,
        narrator_id=None,
      )  # TODO: Modify narrator_id and book_id
    )

  # Process books in parallel
  with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    results = list(
      tqdm(
        executor.map(book_alignment, books),
        total=len(books),
        desc="Processing books",
        leave=True,
        dynamic_ncols=True,
      )
    )

  successful_books = sum(1 for r in results if r)
  logger.success(
    f"Alignment completed. {successful_books}/{len(books)} books processed successfully."
  )


if __name__ == "__main__":
  main()
