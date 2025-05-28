import glob
import os
import os.path as osp
import shutil
import subprocess
import sys
from typing import List, Tuple, Optional

import pandas as pd
from loguru import logger

from tts_data_pipeline import constants

logger.remove()
logger.add(
  f"{constants.LOG_DIR}/pre_processing.log",
  level="INFO",
  rotation="10 MB",
  encoding="utf-8",
  format=constants.FORMAT_LOG,
  colorize=False,
  diagnose=True,
  enqueue=True,
)


def check_ffmpeg():
  if not shutil.which("ffmpeg"):
    logger.error("ffmpeg is not installed or not in PATH. Please install ffmpeg first.")
    return False

  return True


# CAUTION: Not used
def delete_old_processed_audio():
  shutil.rmtree(constants.AUDIO_QUALIFIED_DIR)
  shutil.rmtree(constants.AUDIO_UNQUALIFIED_DIR)


def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
  """
  Convert an MP3 file to WAV format using ffmpeg.

  Args:
      mp3_path (str): Path to the MP3 file
      wav_path (str): Path where the WAV file will be saved

  Returns:
      bool: True if conversion is successful, False otherwise
  """
  # Check if ffmpeg is installed
  if not check_ffmpeg():
    return False

  # Convert MP3 to WAV by ffmpeg
  os.makedirs(osp.dirname(wav_path), exist_ok=True)
  try:
    subprocess.run(
      ["ffmpeg", "-y", "-i", mp3_path, wav_path],
      check=True,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )
    logger.success(f"Converted {mp3_path} to {wav_path}")
    return True
  except subprocess.CalledProcessError as e:
    logger.exception(f"ffmpeg failed for {mp3_path}: {e}")
    return False
  except Exception as e:
    logger.exception(f"Error converting {mp3_path} to WAV: {e}")
    return False


def get_sample_rate(mp3_path: str) -> int:
  """
  Get the sample rate of an MP3 file using ffprobe.

  Args:
      mp3_path (str): Path to the MP3 file.

  Returns:
      int: Sample rate in Hz, or 0 if there was an error.
  """
  if not check_ffmpeg():
    return 0

  try:
    result = subprocess.run(
      [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        mp3_path,
      ],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
    )
    return int(result.stdout.strip()) if result.returncode == 0 else 0
  except Exception as e:
    logger.error(f"Error getting sample rate for {mp3_path}: {e}")
    sys.exit(1)


def combine_wav_files(output_path: str, input_paths: List[str]):
  """
  Combine multiple WAV files into a single WAV file using ffmpeg concat demuxer.

  Args:
      output_path (str): Path to save the combined WAV file.
      input_paths (List[str]): List of WAV file paths to combine.
  """
  if not check_ffmpeg():
    return

  if not input_paths:
    logger.warning("No input WAV files provided for combination.")
    return

  input_paths = sorted(input_paths)

  try:
    # Create a temporary text file listing input WAV files
    list_file = osp.join(osp.dirname(output_path), "input_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
      for path in input_paths:
        f.write(f"file '{osp.abspath(path)}'\n")

    # Use ffmpeg with concat demuxer
    subprocess.run(
      [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-c",
        "copy",
        output_path,
      ],
      check=True,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )

    # Clean up the temporary list file
    os.remove(list_file)

    logger.info(
      f"Successfully combined {len(input_paths)} WAV files into {output_path}"
    )

  except subprocess.CalledProcessError as e:
    logger.error(f"ffmpeg failed to combine files: {e}")
  except Exception as e:
    logger.error(f"Unexpected error combining WAV files: {e}")


def get_audio_duration(audio_path: str) -> float:
  """
  Get the duration of an audio file in seconds.

  Args:
      audio_path (str): Path to the audio file.

  Returns:
      float: Duration of the audio file in seconds.
  """
  if (
    not check_ffmpeg()
  ):  # Because ffprobe is installed with ffmpeg, checking ffmpeg is enough
    return 0

  try:
    cmd = [
      "ffprobe",
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      audio_path,
    ]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return float(output.decode("utf-8").strip())
  except subprocess.CalledProcessError as e:
    logger.exception(f"ffprobe failed to get duration: {e}")
    return 0


# CAUTION: Not used
def split_audiobook(
  book_name: str,
  input_audio_paths: List[str],
  time_threshold: int = 1800,  # 30 minutes
  convert_to_wav: bool = True,
) -> Optional[Tuple[List[str], int]]:
  """
  Split an audiobook into parts based on the minute threshold.

  Args:
    book_name (str): Name of the audiobook.
    input_audio_paths (List[str]): Path to all audio parts of the audiobook.
    time_threshold (int): The time threshold (in seconds) for splitting the audiobook.

  Returns:
      List[str]: List of paths to the split audio files.
  """
  # Check if ffmpeg is installed
  if not check_ffmpeg():
    return None

  # Split each audiobook's part
  audio_name_parts = [
    osp.splitext(osp.basename(audio_path))[0] for audio_path in input_audio_paths
  ]
  # Temporary split directory
  split_dirs = [
    osp.join(constants.AUDIO_QUALIFIED_DIR, book_name, audio_part)
    for audio_part in audio_name_parts
  ]  # e.g. constants.AUDIO_QUALIFIED_DIR/bookX/bookX_1, constants.AUDIO_QUALIFIED_DIR/bookX/bookX_2, ...
  for input_audio_path, audio_part, split_dir in zip(
    input_audio_paths, audio_name_parts, split_dirs
  ):
    os.makedirs(split_dir, exist_ok=True)
    try:
      cmd = [
        "ffmpeg",
        "-i",
        input_audio_path,
        "-f",
        "segment",
        "-segment_time",
        str(time_threshold),
        "-c",
        "copy",
        osp.join(
          split_dir, f"{audio_part}_%03d.mp3"
        ),  # e.g. abc_1_001.mp3, abc_1_002.mp3, etc.
      ]
      subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
      )
      logger.info(
        f"Successfully split {input_audio_path} into {time_threshold} seconds, total {len(os.listdir(split_dir))} files."
      )
    except Exception as e:
      logger.exception(f"Error splitting {input_audio_path}: {e}")
      return None

  # Remark all split files
  new_audio_dir = osp.join(constants.AUDIO_QUALIFIED_DIR, book_name)

  counter = 1
  for file in glob.glob(new_audio_dir + "/*/*.mp3"):
    # If convert_to_wav is True, convert mp3 to wav, else save splitted mp3
    if convert_to_wav:
      convert_mp3_to_wav(file, osp.join(new_audio_dir, f"{book_name}_{counter}.wav"))
      os.remove(file)  # Remove mp3 file after converting to wav
    else:
      shutil.move(file, osp.join(new_audio_dir, f"{book_name}_{counter}.mp3"))
    counter += 1

  # Remove the temporary directories
  for audio_dir in split_dirs:
    shutil.rmtree(audio_dir)

  return glob.glob(new_audio_dir + "/*.mp3"), counter


def audio_processing(
  mp3_paths: List[str], update_metadata: bool = True, remove_original_files: bool = True
):
  """
  Process a single audio file. It contains converting MP3 to WAV and checking sample rate.

  Args:
      update_metadata (bool, optional): Whether to update metadata file. Defaults to False.
  """

  # Read the metadata file for updating sample rate
  metadata_df = (
    pd.read_csv(constants.METADATA_BOOK_PATH) if update_metadata else pd.DataFrame()
  )

  # Get audiobook name
  audiobook_name = osp.basename(mp3_paths[0]).split("_")[0]

  # Convert mp3 to wav
  wav_paths = []
  for mp3_path in mp3_paths:
    wav_path = osp.join(
      constants.AUDIO_QUALIFIED_DIR, osp.basename(mp3_path).replace(".mp3", ".wav")
    )
    wav_paths.append(wav_path)
    convert_mp3_to_wav(mp3_path, wav_path)

  # Combine mp3 files into a single wav file
  output_wav_path = osp.join(constants.AUDIO_QUALIFIED_DIR, f"{audiobook_name}.wav")
  combine_wav_files(output_wav_path, wav_paths)

  # Check sample rate
  sample_rate = get_sample_rate(output_wav_path)

  if sample_rate < constants.MIN_SAMPLE_RATE:
    logger.error(
      f"Sample rate for {audiobook_name}, which is {sample_rate}, is less than {constants.MIN_SAMPLE_RATE}"
    )
    shutil.move(output_wav_path, constants.AUDIO_UNQUALIFIED_DIR)
    # Update qualified column
    metadata_df.loc[
      metadata_df["audio_url"].str.contains(audiobook_name), "qualified"
    ] = 0
  else:
    logger.info(f"Sample rate for {audiobook_name} is {sample_rate}")
    # Update qualified column
    metadata_df.loc[
      metadata_df["audio_url"].str.contains(audiobook_name), "qualified"
    ] = 1

  # Update sample rate column for metadata
  if update_metadata:
    metadata_df.loc[
      metadata_df["audio_url"].str.contains(audiobook_name), "sample_rate"
    ] = sample_rate

  # Save metadata
  if update_metadata:
    metadata_df.to_csv(constants.METADATA_BOOK_PATH, index=False)

  # Remove the original files
  if remove_original_files:
    [os.remove(mp3_path) for mp3_path in mp3_paths]

  return audiobook_name
