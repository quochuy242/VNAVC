import glob
import os
import os.path as osp
import shutil
from typing import List, Optional, Tuple
import subprocess
import av
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


def convert_audio_to_wav(input_path: str, wav_path: str) -> bool:
  """
  Convert an audio file to WAV format using PyAV.
  Args:
      input_path (str): Path to the input audio file
      wav_path (str): Path where the WAV file will be saved
  Returns:
      bool: True if conversion is successful, False otherwise
  """

  os.makedirs(osp.dirname(wav_path), exist_ok=True)

  try:
    # Open input container
    input_container = av.open(input_path)

    # Get the first audio stream
    audio_stream = input_container.streams.audio[0]

    # Open output container
    output_container = av.open(wav_path, "w")

    # Add audio stream to output with proper channel configuration
    output_stream = output_container.add_stream(
      "pcm_s16le", rate=audio_stream.sample_rate, layout=audio_stream.layout
    )

    # Process audio frames
    for frame in input_container.decode(audio_stream):
      for packet in output_stream.encode(frame):
        output_container.mux(packet)

    # Flush encoder
    for packet in output_stream.encode():
      output_container.mux(packet)

    # Close containers
    input_container.close()
    output_container.close()

    logger.success(f"Converted {input_path} to {wav_path}")
    return True

  except Exception as e:
    logger.exception(f"Error converting {input_path} to WAV: {e}")
    return False


def get_sample_rate(audio_path: str) -> int:
  """
  Get the sample rate of an audio file using PyAV.
  Args:
      audio_path (str): Path to the audio file.
  Returns:
      int: Sample rate in Hz, or 0 if there was an error.
  """

  try:
    container = av.open(audio_path)
    audio_stream = container.streams.audio[0]
    sample_rate = audio_stream.sample_rate
    container.close()
    return sample_rate

  except Exception as e:
    logger.error(f"Error getting sample rate for {audio_path}: {e}")
    return 0


def combine_wav_files(output_path: str, input_paths: List[str]):
  """
  Combine multiple WAV files into a single WAV file using PyAV.
  Args:
      output_path (str): Path to save the combined WAV file.
      input_paths (List[str]): List of WAV file paths to combine.
  """

  if not input_paths:
    logger.warning("No input WAV files provided for combination.")
    return

  input_paths = sorted(input_paths)

  try:
    # Get audio properties from first file
    first_container = av.open(input_paths[0])
    first_stream = first_container.streams.audio[0]
    sample_rate = first_stream.sample_rate
    layout = first_stream.layout
    first_container.close()

    # Create output container
    output_container = av.open(output_path, "w")
    output_stream = output_container.add_stream(
      "pcm_s16le", rate=sample_rate, layout=layout
    )

    # Process each input file
    for input_path in input_paths:
      input_container = av.open(input_path)
      audio_stream = input_container.streams.audio[0]

      # Decode and encode frames
      for frame in input_container.decode(audio_stream):
        for packet in output_stream.encode(frame):
          output_container.mux(packet)

      input_container.close()

    # Flush encoder
    for packet in output_stream.encode():
      output_container.mux(packet)

    output_container.close()

    logger.info(
      f"Successfully combined {len(input_paths)} WAV files into {output_path}"
    )

  except Exception as e:
    logger.error(f"Error combining WAV files: {e}")


def get_audio_duration(audio_path: str) -> float:
  """
  Get the duration of an audio file in seconds using PyAV.
  Args:
      audio_path (str): Path to the audio file.
  Returns:
      float: Duration of the audio file in seconds.
  """

  try:
    container = av.open(audio_path)
    duration = float(container.duration) / av.time_base if container.duration else 0
    container.close()
    return duration

  except Exception as e:
    logger.exception(f"Error getting duration for {audio_path}: {e}")
    return 0


def audio_processing(
  input_paths: List[str],
  update_metadata: bool = True,
  remove_original_files: bool = True,
):
  """
  Process a single audio file. It contains converting audio to WAV and checking sample rate.
  Args:
      input_paths (List[str]): List of input audio file paths
      update_metadata (bool, optional): Whether to update metadata file. Defaults to True.
      remove_original_files (bool, optional): Whether to remove original files. Defaults to True.
  """
  # Read the metadata file for updating sample rate
  metadata_df = (
    pd.read_csv(constants.METADATA_BOOK_PATH) if update_metadata else pd.DataFrame()
  )

  # Get audiobook name
  audiobook_name = osp.basename(input_paths[0]).split("_")[0]

  # Convert audio files to wav
  wav_paths = []
  for input_path in input_paths:
    # Get the base name and replace extension with .wav
    base_name = osp.splitext(osp.basename(input_path))[0]
    wav_path = osp.join(constants.AUDIO_QUALIFIED_DIR, f"{base_name}.wav")
    wav_paths.append(wav_path)
    convert_audio_to_wav(input_path, wav_path)

  # Combine audio files into a single wav file
  output_wav_path = osp.join(constants.AUDIO_QUALIFIED_DIR, f"{audiobook_name}.wav")
  if len(wav_paths) == 1:
    shutil.move(wav_paths[0], output_wav_path)
  else:
    combine_wav_files(output_wav_path, wav_paths)
    # Clean up individual WAV files after combining
    for wav_path in wav_paths:
      if osp.exists(wav_path):
        os.remove(wav_path)

  # Check sample rate
  sample_rate = get_sample_rate(output_wav_path)
  if sample_rate < constants.MIN_SAMPLE_RATE:
    logger.error(
      f"Sample rate for {audiobook_name}, which is {sample_rate}, is less than {constants.MIN_SAMPLE_RATE}"
    )
    shutil.move(output_wav_path, constants.AUDIO_UNQUALIFIED_DIR)
    # Update qualified column
    if update_metadata:
      metadata_df.loc[
        metadata_df["audio_url"].str.contains(audiobook_name), "qualified"
      ] = 0
  else:
    logger.info(f"Sample rate for {audiobook_name} is {sample_rate}")
    # Update qualified column
    if update_metadata:
      metadata_df.loc[
        metadata_df["audio_url"].str.contains(audiobook_name), "qualified"
      ] = 1

  # Update sample rate column for metadata
  if update_metadata:
    metadata_df.loc[
      metadata_df["audio_url"].str.contains(audiobook_name), "sample_rate"
    ] = sample_rate
    metadata_df.loc[
      metadata_df["audio_url"].str.contains(audiobook_name), "audio_size"
    ] = os.path.getsize(output_wav_path) if osp.exists(output_wav_path) else 0

  # Save metadata
  if update_metadata:
    metadata_df.to_csv(constants.METADATA_BOOK_PATH, index=False)

  # Remove the original files
  if remove_original_files:
    for input_path in input_paths:
      if osp.exists(input_path):
        os.remove(input_path)

  return audiobook_name


# CAUTION: Not used
def check_ffmpeg():
  if not shutil.which("ffmpeg"):
    logger.error("ffmpeg is not installed or not in PATH. Please install ffmpeg first.")
    return False
  return True


# CAUTION: Not used
def delete_old_processed_audio():
  shutil.rmtree(constants.AUDIO_QUALIFIED_DIR)
  shutil.rmtree(constants.AUDIO_UNQUALIFIED_DIR)


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
      subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
      convert_audio_to_wav(file, osp.join(new_audio_dir, f"{book_name}_{counter}.wav"))
      os.remove(file)  # Remove mp3 file after converting to wav
    else:
      shutil.move(file, osp.join(new_audio_dir, f"{book_name}_{counter}.mp3"))
    counter += 1

  # Remove the temporary directories
  for audio_dir in split_dirs:
    shutil.rmtree(audio_dir)

  return glob.glob(new_audio_dir + "/*.mp3"), counter
