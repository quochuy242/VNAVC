import os
import os.path as osp
import subprocess
from typing import Optional

import pandas as pd

from tts_data_pipeline import constants


# TODO: Check the logic of the func built by AI again
def split_audio(
  alignment_path: str, audio_path: str, speaker_id: Optional[int]
) -> None:
  """
  Split audio file based on alignment data.

  Args:
      alignment_path (str): Path to the TSV alignment file
      audio_path (str): Path to the source audio file
      speaker_id (str, optional): Speaker ID. Defaults to None.
  """
  if speaker_id is None:
    output_dir = os.path.join(constants.DATASET_DIR, "audio_segments")
  else:
    output_dir = ...  # TODO: Modify output_dir
  os.makedirs(output_dir, exist_ok=True)

  # Read alignment data
  align_df = pd.read_csv(
    alignment_path, sep="\t", names=["start", "end", "id", "duration"]
  )

  # Get base filename without extension
  base_name = os.path.basename(audio_path).rsplit(".", 1)[0]

  # Split audio for each segment
  for idx, row in align_df.iterrows():
    output_file = os.path.join(output_dir, f"{base_name}_{idx:04d}.wav")

    cmd = [
      "ffmpeg",
      "-y",  # Overwrite output files without asking
      "-i",
      audio_path,
      "-ss",
      str(row["start"]),
      "-to",
      str(row["end"]),
      "-c:a",
      "pcm_s16le",  # Use uncompressed audio for better quality
    ]

    # Resample to standard sample rate
    if not constants.STANDARD_SAMPLE_RATE:
      cmd.extend(["-ar", str(constants.STANDARD_SAMPLE_RATE)])

    cmd.append(output_file)

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# TODO: Check the logic of the func built by AI again
def split_text(
  alignment_path: str, text_path: str, speaker_id: Optional[int] = None
) -> None:
  """
  Split text file based on alignment data.

  Args:
      alignment_path (str): Path to the TSV alignment file
      text_path (str): Path to the source text file
      speaker_id (str, optional): Speaker ID. Defaults to None.
  """
  if speaker_id is None:
    output_dir = os.path.join(constants.DATASET_DIR, "text_segments")
  else:
    output_dir = ...  # TODO: Modify output_dir
  os.makedirs(output_dir, exist_ok=True)

  # Read alignment data
  align_df = pd.read_csv(alignment_path, sep="\t", names=["start", "end", "id"])

  # Read all text lines
  with open(text_path, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

  # Get base filename without extension
  base_name = os.path.basename(text_path).rsplit(".", 1)[0]

  # Save each line to a separate file
  for idx, line in enumerate(lines):
    if idx >= len(align_df):
      break

    output_file = os.path.join(output_dir, f"{base_name}_{idx:04d}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
      f.write(line)
