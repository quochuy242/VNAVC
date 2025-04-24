import os
import shutil
import pandas as pd
import subprocess

from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from loguru import logger

from tts_data_pipeline import constants

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


def align_audio_text(audio_path: str, text_path: str, output_path: str) -> None:
  """
  Align a single audio and text file using Aeneas and save the output syncmap.

  Args:
      audio_path (str): Path to the audio file.
      text_path (str): Path to the text file.
      output_path (str): Path to save the syncmap (JSON format).
  """
  # Check dependencies
  if not check_dependencies():
    return

  # Run aeneas
  try:
    task = Task(config_string=constants.AENEAS_CONFIG)
    task.audio_file_path_absolute = os.path.abspath(audio_path)
    task.text_file_path_absolute = os.path.abspath(text_path)
    task.sync_map_file_path_absolute = os.path.abspath(output_path)

    ExecuteTask(task).execute()
    task.output_sync_map_file()
  except Exception as e:
    logger.error(f"Aeneas failed for {audio_path}: {e}")

  # Post-processing
  process_aligment_output(output_path, text_path)

  # Splitting
  split_audio(output_path, audio_path)
  split_text(output_path, text_path)


def check_dependencies():
  """Check if the required dependencies of aeneas are installed."""
  deps = ["ffmpeg", "ffprobe", "espeak"]

  for dep in deps:
    if not shutil.which(dep):
      logger.error(f"{dep} is not installed or not in PATH.")
      return False

  return True


def process_aligment_output(alignment_path: str, text_path: str) -> None:
  """
  Process the alignment output

  Args:
      output_path (str): Path to output file from aeneas
  """

  def remove_outliers(
    df: pd.DataFrame, column: str, z_threshold: float = 3.0
  ) -> pd.DataFrame:
    # Calc z-score
    df["z_score"] = (df[column] - df[column].mean()) / df[column].std()

    # Remove outliers
    outlier_idxs = df["z_score"].abs() <= z_threshold

    logger.info(f"The length of outliers: {len(outlier_idxs)}")
    return outlier_idxs

  align_df = pd.read_csv(alignment_path, sep="\t", names=["start", "end", "id"])
  align_df["duration"] = align_df["end"] - align_df["start"]

  # Remove first and last sentences because that origin sentence is splitted
  drop_idxs = [0, len(align_df) - 1]

  # Remove outliers
  drop_idxs.extend(remove_outliers(align_df, "duration", z_threshold=3.0))

  # Load the text source
  with open(text_path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

  # Update the alignment output and the text source
  drop_idxs = set(drop_idxs)
  align_df = align_df.drop(index=drop_idxs)
  lines = [item for idx, item in enumerate(lines) if idx not in drop_idxs]
  align_df.to_csv(alignment_path)
  with open(text_path, "w", encoding="utf-8") as f:
    for line in lines:
      f.write(line + "\n")


def split_audio(
  alignment_path: str, audio_path: str, output_path: str = constants.DATASET_DIR
):
  # TODO: Modify below variables
  start_time = 0
  end_time = 0
  output_file = None

  cmd = [
    "ffmpeg",
    "-i",
    audio_path,
    "-ss",
    str(start_time),
    "-to",
    str(end_time),
    "-c",
    "copy",
    output_file,
  ]
  subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def split_text(
  alignment_path: str, text_path: str, output_path: str = constants.DATASET_DIR
): ...
