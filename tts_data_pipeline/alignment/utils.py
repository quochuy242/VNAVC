import os
import os.path as osp
import shutil

import numpy as np
import pandas as pd
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from loguru import logger

from tts_data_pipeline.alignment.split import split_audio, split_text
from tts_data_pipeline import constants

from dataclasses import dataclass

# Configure logger
logger.remove()
logger.add(
  f"{constants.LOG_DIR}/alignment.log",
  level="INFO",
  rotation="10 MB",
  encoding="utf-8",
  format=constants.FORMAT_LOG,
  colorize=False,
  diagnose=True,
  enqueue=True,
)


class BookInfo(dataclass):
  bookname: str
  audio_path: str
  text_path: str
  narrator_id: int


def check_dependencies():
  """Check if the required dependencies of aeneas are installed."""
  deps = ["ffmpeg", "ffprobe", "espeak"]
  missing = [dep for dep in deps if not shutil.which(dep)]

  if missing:
    logger.error(f"Missing dependencies: {', '.join(missing)}. Please install them.")
    return False
  return True


def process_alignment_output(
  alignment_path: str, text_path: str, overwrite: bool = False
) -> None:
  """
  Process the alignment output: remove outliers and update both alignment and text files.

  Args:
      alignment_path (str): Path to output file from aeneas
      text_path (str): Path to the text file
      overwrite (bool, optional): If True, overwrite the output files. Defaults to False.
  """
  # Load the alignment data
  align_df = pd.read_csv(alignment_path, sep="\t", names=["start", "end", "id"])
  align_df["duration"] = align_df["end"] - align_df["start"]

  # Initialize indices to drop
  drop_idxs = [0, len(align_df) - 1]  # Remove first and last sentences (splitted)

  # Remove outliers
  if len(align_df) > 3:  # Need at least a few points to calculate outliers
    z_scores = np.abs(
      (align_df["duration"] - align_df["duration"].mean()) / align_df["duration"].std()
    )
    outlier_idxs = np.where(z_scores > 3.0)[0]
    drop_idxs.extend(outlier_idxs)
    logger.info(f"Found {len(outlier_idxs)} outliers in alignment")

  # Load the text source
  with open(text_path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

  # // Remove chapter headers (e.g., "CHƯƠNG XY")
  # // chapter_indices = [
  # //   i for i, line in enumerate(lines) if line.strip().startswith("CHƯƠNG")
  # // ]
  # // drop_idxs.extend(chapter_indices)

  # Update the alignment output and the text source
  drop_idxs = set(drop_idxs)
  align_df_filtered = align_df.drop(index=[i for i in drop_idxs if i < len(align_df)])

  # Get only the lines that we're keeping
  lines_filtered = [line for idx, line in enumerate(lines) if idx not in drop_idxs]

  # Write updated alignment
  align_df_filtered.to_csv(alignment_path, sep="\t", header=False, index=False)

  # Write updated text
  if overwrite:
    with open(text_path, "w", encoding="utf-8") as f:
      f.write("\n".join(lines_filtered))
  else:
    new_text_path = text_path.replace("sentence", "filtered_sentence")
    with open(new_text_path, "w", encoding="utf-8") as f:
      f.write("\n".join(lines_filtered))


def align_audio_text(
  audio_path: str, text_path: str, output_path: str, speaker_id: int
) -> bool:
  """
  Align a single audio and text file using Aeneas and save the output syncmap.

  Args:
      audio_path (str): Path to the audio file.
      text_path (str): Path to the text file.
      output_path (str): Path to save the syncmap (TSV format).

  Returns:
      bool: True if alignment was successful, False otherwise.
  """
  # Check dependencies only once at the beginning of the program
  try:
    task = Task(config_string=constants.AENEAS_CONFIG)
    task.audio_file_path_absolute = os.path.abspath(audio_path)
    task.text_file_path_absolute = os.path.abspath(text_path)
    task.sync_map_file_path_absolute = os.path.abspath(output_path)

    ExecuteTask(task).execute()
    task.output_sync_map_file()

    # Process the alignment output and split the files
    process_alignment_output(output_path, text_path)
    split_audio(output_path, audio_path, speaker_id)
    split_text(output_path, text_path, speaker_id)

    return True
  except Exception as e:
    logger.error(f"Aeneas failed for {audio_path}: {e}")
    return False


def book_alignment(book_info: BookInfo) -> bool:
  """Process alignment for a single book with multiple parts"""
  bookname, audio_path, text_path, speaker_id = (
    book_info.bookname,
    book_info.audio_path,
    book_info.text_path,
    book_info.narrator_id,
  )

  os.makedirs(osp.join(constants.AENEAS_OUTPUT_DIR, bookname), exist_ok=True)
  audio_parts = sorted(os.listdir(audio_path))
  text_parts = sorted(os.listdir(text_path))

  if len(audio_parts) != len(text_parts):
    logger.error(
      f"For book {bookname}: The length of audio and text parts do not match ({len(audio_parts)=} != {len(text_parts)=})"
    )
    return False

  success_count = 0
  for audio_part, text_part in zip(audio_parts, text_parts):
    audio_part_path = osp.join(audio_path, audio_part)
    text_part_path = osp.join(text_path, text_part)
    output_path = osp.join(
      constants.AENEAS_OUTPUT_DIR,
      bookname,
      audio_part.replace(".wav", ".tsv"),
    )

    if align_audio_text(audio_part_path, text_part_path, output_path, speaker_id):
      success_count += 1

  logger.success(
    f"Book {bookname}: {success_count}/{len(audio_parts)} parts aligned successfully"
  )
  return success_count == len(audio_parts)
