import argparse
import os

from tts_data_pipeline import constants
from tts_data_pipeline.pre_processing.audio import audio_processing, group_audiobook
from tts_data_pipeline.pre_processing.text import text_processing
from tts_data_pipeline.pre_processing.utils import logger


def process_text_file(pdf_file): ...


def process_audio_file(audio_file): ...


def main():
  parser = argparse.ArgumentParser(
    description="Pre-processing a single text or audio file."
  )
  parser.add_argument(
    "-t",
    "--type",
    choices=["text", "audio", "all"],
    help="Type of file to process or process all files.",
  )
  parser.add_argument("-p", "--path", help="Path to the file to process.")
  args = parser.parse_args()

  if args.type == "text":
    process_text_file(args.path)
  elif args.type == "audio":
    process_audio_file(args.path)


if __name__ == "__main__":
  main()
