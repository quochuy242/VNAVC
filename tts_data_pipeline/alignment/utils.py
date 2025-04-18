import json
import os
import shutil
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
  logger.info(f"Aligning {audio_path} with {text_path}")
  task = Task(config_string=constants.AENEAS_CONFIG)
  task.audio_file_path_absolute = os.path.abspath(audio_path)
  task.text_file_path_absolute = os.path.abspath(text_path)
  task.sync_map_file_path_absolute = os.path.abspath(output_path)

  ExecuteTask(task).execute()
  task.output_sync_map_file()


def check_dependencies():
  """Check if the required dependencies of aeneas are installed."""
  deps = ["ffmpeg", "ffprobe", "espeak-ng"]

  for dep in deps:
    if not shutil.which(dep):
      logger.error(f"{dep} is not installed or not in PATH.")
      return False

  return True


def process_json_output(output_path: str) -> None:
  """
  Process the JSON output file created by Aeneas.

  Args:
      output_path (str): Path to the JSON output file.
  """
  with open(output_path, "r", encoding="utf-8") as f:
    output = json.load(f)

  for key, value in output.items():
    ...
