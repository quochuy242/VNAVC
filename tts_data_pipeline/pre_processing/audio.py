import glob
import os
import os.path as osp
import shutil
import subprocess
import sys
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from tts_data_pipeline import constants

logger.remove()
logger.add(
  f"{constants.LOG_DIR}/audio_processing.log",
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
    logger.error(f"ffmpeg failed for {mp3_path}: {e}")
    return False
  except Exception as e:
    logger.error(f"Error converting {mp3_path} to WAV: {e}")
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


def group_audiobook(mp3_dir: str, unqualified_dir: str) -> List[List[str]]:
  """Efficiently group all parts of audiobooks based on file name prefix.

  Args:
      mp3_dir (str): Path to directory containing mp3 files.
      unqualified_dir (str): Directory to move unqualified files

  Returns:
      List[List[str]]: List of lists, where each sublist contains mp3 file paths of the same audiobook.
  """
  grouped = defaultdict(list)

  for mp3_file in os.listdir(mp3_dir):
    # If the file is a directory, skip it
    if osp.isdir(osp.join(mp3_dir, mp3_file)):
      continue

    file_path = osp.join(mp3_dir, mp3_file)

    # If the file is not an MP3 file, move it to unqualified folder
    if not mp3_file.endswith(".mp3"):
      logger.warning(
        f"File {mp3_file} is not an MP3 file, move it to {unqualified_dir}"
      )
      shutil.move(file_path, unqualified_dir)
      continue

    book_name = mp3_file.split("_")[0]
    grouped[book_name].append(file_path)

  return [sorted(files) for files in grouped.values()]


# NOTE: Not used
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

    # Remove the input WAV files
    [os.remove(path) for path in input_paths]
    logger.info(f"Removed input WAV files: {', '.join(input_paths)}")

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
    logger.error(f"ffprobe failed to get duration: {e}")
    return 0


def split_audiobook(
  book_name: str,
  input_audio_paths: List[str],
  time_threshold: int = 1800,  # 30 minutes
  convert_to_wav: bool = True,
) -> List[str]:
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
    return []

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
      logger.error(f"Error splitting {input_audio_path}: {e}")
      return []

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


def process_audio_files(
  update_metadata: bool = False,
):
  """
  Process all MP3 files in a directory:
  1. Group all parts of an audiobook together
  2. Convert them to WAV format
  3. Check their sample rate
  4. Move those with sample rates below the threshold to unqualified folder
  """
  # Create output directories if they don't exist
  os.makedirs(constants.AUDIO_QUALIFIED_DIR, exist_ok=True)
  os.makedirs(constants.AUDIO_UNQUALIFIED_DIR, exist_ok=True)

  # Read the metadata file for updating sample rate
  if update_metadata:
    metadata_df = pd.read_csv(constants.METADATA_BOOK_PATH)
    metadata_df["sample_rate"], metadata_df["qualified"], metadata_df["num_parts"] = (
      pd.Series([None] * len(metadata_df)),  # A new column for sample rate
      pd.Series([1] * len(metadata_df)),  # A new column for qualified
      pd.Series([1] * len(metadata_df)),  # A new column for num_parts
    )

  # Get all MP3 file paths in the audio directory
  audiobooks = group_audiobook(
    constants.AUDIO_RAW_DIR, unqualified_dir=constants.AUDIO_UNQUALIFIED_DIR
  )

  # Check no MP3 files
  if not audiobooks:
    logger.warning(f"No MP3 files found in {constants.AUDIO_RAW_DIR}")
    return

  # Process each MP3 file
  qualified_count = 0
  unqualified_count = 0
  for audiobook in tqdm(audiobooks, desc="Processing audio files"):
    # Get sample rates for each MP3 file
    sample_rates = [get_sample_rate(mp3_path) for mp3_path in audiobook]
    min_sample_rate = min(sample_rates)

    # Get audiobook name
    audiobook_name = osp.basename(audiobook[0]).split("_")[0]

    # Check sample rates to determine the quality's book
    if min_sample_rate < constants.MIN_SAMPLE_RATE:
      unqualified_count += 1
      logger.warning(
        f"Unqualified book: {audiobook_name}, minimum sample rates: {min_sample_rate}"
      )
      # Move unqualified files to unqualified folder
      for mp3_path in audiobook:
        shutil.move(mp3_path, constants.AUDIO_UNQUALIFIED_DIR)

        # Update qualified column for metadata
        if update_metadata:
          metadata_df.loc[
            metadata_df["audio_url"].str.contains(mp3_path.split("/")[-1]),
            "qualified",
          ] = 0
    else:
      qualified_count += 1
      logger.info(
        f"Qualified book: {audiobook_name}, minimum sample rate: {min_sample_rate}"
      )

      # Split audiobook into parts, default time threshold is 30 minutes
      split_paths, num_parts = split_audiobook(
        audiobook_name, audiobook, convert_to_wav=True
      )

      # Update sample rate column for metadata
      if update_metadata:
        metadata_df.loc[
          metadata_df["audio_url"].str.contains(audiobook_name), "sample_rate"
        ] = np.mean(sample_rates)
        metadata_df.loc[
          metadata_df["audio_url"].str.contains(audiobook_name), "num_parts"
        ] = num_parts

  # Save metadata
  if update_metadata:
    metadata_df.to_csv(constants.METADATA_BOOK_PATH, index=False)

  logger.info(
    f"Processing complete: \n - Total books processed: {len(audiobooks)}\n - Qualified books (≥ {constants.MIN_SAMPLE_RATE} Hz): {qualified_count}\n - Unqualified books (< {constants.MIN_SAMPLE_RATE} Hz): {unqualified_count}"
  )


def delete_old_processed_audio():
  shutil.rmtree(constants.AUDIO_QUALIFIED_DIR)
  shutil.rmtree(constants.AUDIO_UNQUALIFIED_DIR)


if __name__ == "__main__":
  process_audio_files(update_metadata=True)
  logger.success("Audio processing complete")
