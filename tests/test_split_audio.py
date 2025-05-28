import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tts_data_pipeline import Book, Narrator, constants
from tts_data_pipeline.alignment.utils import split_audio


def old_split_audio(book: Book, max_workers: int = 8) -> None:
  """
  Split audio file based on alignment data using multithreading.
  """
  if isinstance(book.narrator, Narrator):
    output_dir = Path(constants.DATASET_DIR) / (
      str(book.narrator.id) if book.narrator.id else book.narrator.name
    )
  elif isinstance(book.narrator, list):
    output_dir = Path(constants.DATASET_DIR) / (
      str(book.narrator[0].id) if book.narrator[0].id else book.narrator[0].name
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
        desc="Old splitting Audio",
      )
    )


def main():
  book = Book.from_json("./data/metadata/book/mat-biec.json")
  book.update_paths(
    audio_path="./data/audio/qualified/mat-biec.wav",
    alignment_path="./data/alignment/mat-biec/output.tsv",
  )

  # Benchmark split_audio
  print("Benchmarking split_audio...")

  # Test result: 3.98 seconds
  start_time = time.time()
  split_audio(book)
  print(f"New split_audio: {time.time() - start_time:.2f} seconds")

  # Test result: 56 minutes
  # start_time = time.time()
  # old_split_audio(book, max_workers=8)
  # print(f"Old split_audio: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
  main()
