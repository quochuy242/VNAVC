import argparse
import os
import os.path as osp
import subprocess
from typing import List

from tts_data_pipeline import constants
from tts_data_pipeline.pre_processing.audio import (
  audio_processing,
)
from tts_data_pipeline.pre_processing.audio import (
  logger as audio_logger,
)
from tts_data_pipeline.pre_processing.text import (
  logger as text_logger,
)
from tts_data_pipeline.pre_processing.text import (
  text_processing,
)


def process_text_file(pdf_path: str, **kwargs):
  try:
    pdf_filename = text_processing(pdf_path, **kwargs)
    text_logger.success(f"Processing {pdf_filename} pdf file completed")
  except Exception as e:
    text_logger.exception(f"Error processing {pdf_path}: {e}")


def process_audio_file(audio_paths: List[str], **kwargs):
  """
  Args:
      audio_paths (List[str]): The WAV file which is full audiobook
  """
  try:
    audiobook_name = audio_processing(audio_paths, **kwargs)
    audio_logger.success(f"Audiobook {audiobook_name} processing completed")
  except Exception as e:
    audio_logger.exception(f"Error processing {audio_paths[0]}: {e}")


def main():
  parser = argparse.ArgumentParser(
    description="Pre-processing a single text or audio file."
  )
  parser.add_argument(
    "-t",
    "--target",
    choices=["text", "audio", "all"],
    help="Type of file to process or process all files.",
  )
  parser.add_argument(
    "-p",
    "--path",
    help="Path to the file with text or the directory with audio to process.",
  )
  args = parser.parse_args()

  if args.target == "text":
    process_text_file(
      args.path,
      min_word_threshold=constants.MIN_WORD_THRESHOLD,
      remove_original_files=False,
    )
  elif args.target == "audio":
    if not osp.exists(args.path):
      audio_logger.error(f"Path {args.path} does not exist")
      return
    else:
      audio_paths = [
        osp.join(args.path, f)
        for f in os.listdir(args.path)
        if osp.isfile(osp.join(args.path, f))
      ]
      process_audio_file(
        audio_paths,
        remove_original_files=False,
      )
  else:
    subprocess.run(["python3", "tts_data_pipeline/pre_processing/process_all.py"])


if __name__ == "__main__":
  main()
