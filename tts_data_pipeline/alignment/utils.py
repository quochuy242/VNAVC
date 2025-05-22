import os
import os.path as osp
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from loguru import logger
from tqdm import tqdm

from tts_data_pipeline import Book, Narrator, constants

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


def check_dependencies():
  """Check if the required dependencies of aeneas are installed."""
  deps = ["ffmpeg", "ffprobe", "espeak"]
  missing = [dep for dep in deps if not shutil.which(dep)]

  if missing:
    logger.error(f"Missing dependencies: {', '.join(missing)}. Please install them.")
    return False
  return True


def split_audio(book: Book, max_workers: int = 8) -> None:
  """
  Split audio file based on alignment data using multithreading.
  """
  if isinstance(book.narrator, Narrator):
    output_dir = Path(constants.DATASET_DIR) / (
      str(book.narrator.id) if book.narrator.id else book.narrator.name
    )
  else:
    output_dir = Path(constants.DATASET_DIR) / "Unknown"
  os.makedirs(output_dir, exist_ok=True)

  align_df = pd.read_csv(
    book.alignment_path, sep="\t", names=["start", "end", "id", "duration"]
  )

  def process_segment(row):
    output_file = output_dir / f"{book.id}_{row['id']}.wav"
    cmd = [
      "ffmpeg",
      "-y",
      "-i",
      book.audio_path,
      "-ss",
      str(row["start"]),
      "-to",
      str(row["end"]),
      "-c:a",
      "pcm_s16le",
    ]
    if constants.STANDARD_SAMPLE_RATE:
      cmd.extend(["-ar", str(constants.STANDARD_SAMPLE_RATE)])
    cmd.append(str(output_file))
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(
      tqdm(
        executor.map(process_segment, [row for _, row in align_df.iterrows()]),
        total=len(align_df),
        desc="Splitting Audio",
      )
    )


def split_text(book: Book, max_workers: int = 8) -> None:
  """
  Split text file based on alignment data using multithreading.
  """
  if isinstance(book.narrator, Narrator):
    output_dir = Path(constants.DATASET_DIR) / (
      str(book.narrator.id) if book.narrator.id else book.narrator.name
    )
  else:
    output_dir = Path(constants.DATASET_DIR) / "Unknown"
  os.makedirs(output_dir, exist_ok=True)

  align_df = pd.read_csv(
    book.alignment_path, sep="\t", names=["start", "end", "id", "duration"]
  )

  if book.text_path is None:
    logger.error("Text path is None")
    return

  with open(book.text_path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

  def process_text_segment(row):
    output_file = output_dir / f"{book.id}_{row['id']}.txt"
    try:
      with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(lines[row["id"]])
    except IndexError:
      logger.warning(f"Line index {row['id']} out of range for {book.text_path}")

  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    list(
      tqdm(
        executor.map(process_text_segment, [row for _, row in align_df.iterrows()]),
        total=len(align_df),
        desc="Splitting Text",
      )
    )


def process_alignment_output(book: Book, remove_first: bool = True) -> None:
  """
  Process the alignment output: remove outliers and update both alignment and text files.

  Args:
      book (Book): Book object containing audio and text paths
      overwrite (bool, optional): If True, overwrite the output files. Defaults to False.
  """
  # Load the alignment data
  align_df = pd.read_csv(book.alignment_path, sep="\t", names=["start", "end", "id"])
  align_df["duration"] = align_df["end"] - align_df["start"]

  # Initialize indices to drop
  drop_idxs = []
  if remove_first:
    drop_idxs.append(0)

  # Remove outliers
  if len(align_df) > 3:
    z_scores = np.abs(
      (align_df["duration"] - align_df["duration"].mean()) / align_df["duration"].std()
    )
    outlier_idxs = np.where(z_scores > 3.0)[0]
    drop_idxs.extend(outlier_idxs)
    logger.info(f"Found {len(outlier_idxs)} outliers in alignment")

  # Update the alignment output and the text source
  drop_idxs = set(drop_idxs)
  align_df_filtered = align_df.drop(index=[i for i in drop_idxs if i < len(align_df)])

  # Write updated alignment
  align_df_filtered.to_csv(book.alignment_path, sep="\t", header=False, index=False)


def book_alignment(book: Book, split: bool, remove_first: bool, jobs: int) -> bool:
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
  os.makedirs(constants.AENEAS_OUTPUT_DIR, exist_ok=True)

  output_path = (
    str(Path(constants.AENEAS_OUTPUT_DIR) / book.name / f"{book.name}.tsv")
    if book.audio_path is not None
    else None
  )

  try:
    if book.alignment_path is not None:
      task = Task(config_string=constants.AENEAS_CONFIG)
      task.audio_file_path_absolute = (
        osp.abspath(book.audio_path) if book.audio_path is not None else None
      )
      task.text_file_path_absolute = (
        osp.abspath(book.text_path) if book.text_path is not None else None
      )
      task.sync_map_file_path_absolute = (
        osp.abspath(output_path) if output_path is not None else None
      )

      ExecuteTask(task).execute()
      task.output_sync_map_file()
      logger.info(f"Alignment output of {book.name} is saved at {output_path}")
      book.update_paths(alignment_path=output_path)
    else:
      logger.warning(f"Alignment file exists: {book.alignment_path}")

    # Process the alignment output and split the files
    if output_path:
      process_alignment_output(book, remove_first=remove_first)
      if split:
        split_audio(book, max_workers=jobs)
        split_text(book, max_workers=jobs)

    logger.success(f"Book {book.name} aligned successfully")
    return True
  except Exception as e:
    logger.error(f"Aeneas failed for {book.audio_path}: {e}")
    return False


def get_size_file(file_path: str) -> int:
  return os.path.getsize(file_path)
