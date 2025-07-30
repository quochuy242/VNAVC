import re
from pathlib import Path
from typing import Optional

import datasets as hf_ds
import librosa
from num2words import num2words
import typer

# Initialize Typer app
app = typer.Typer(
  help="A script to process audio datasets, calculate duration, and push to Hugging Face."
)


def push_to_hf(
  data: list, dataset_id: str, commit: str | None = None, private: bool = False
):
  """Pushes the processed dataset to Hugging Face Hub."""
  dataset = hf_ds.Dataset.from_list(data)
  dataset = dataset.cast_column("audio", hf_ds.Audio(sampling_rate=24000))
  dataset = dataset.class_encode_column("speaker_id")
  dataset.push_to_hub(dataset_id, commit_message=commit, private=private)
  typer.echo(f"Dataset successfully pushed to Hugging Face Hub with commit: '{commit}'")


def process_text(text: str):
  """Cleans and processes text, converting numbers to Vietnamese words."""
  text = text.replace("“", "").replace("”", "")
  text = text.replace("‘", "").replace("’", "")
  text = text.replace("–", "")
  text = text.replace("•", "")
  text = text.replace("…", "")

  text = re.sub(r"\s+", " ", text).strip()

  def replace_number(match):
    number = int(match.group())
    try:
      return num2words(number, lang="vi")
    except Exception as e:
      typer.echo(
        f"Warning: Could not convert number {number} to words. Error: {e}", err=True
      )
      return str(number)

  text = re.sub(r"\b\d+\b", replace_number, text)

  return text


@app.command()
def main(
  # Typer arguments/options
  input_dir: Path = typer.Option(
    "./dataset",
    "--input-dir",
    "-i",
    help="Path to the input dataset directory containing speaker subdirectories.",
    exists=True,
    file_okay=False,
    dir_okay=True,
    readable=True,
    resolve_path=True,
  ),
  create_metadata: bool = typer.Option(
    False,
    "--create-metadata",
    help="Create metadata for dataset.",
  ),
  dataset_id: str = typer.Option(
    "VNAVC",
    "--dataset-id",
    "-d",
    help="Dataset ID on Hugging Face Hub.",
  ),
  commit_message: Optional[str] = typer.Option(
    None,
    "--commit",
    "-c",
    help="Commit message for pushing to Hugging Face Hub. Required if --push-to-hub is True.",
  ),
  private: bool = typer.Option(
    False,
    "--private",
    help="Make the dataset private on Hugging Face Hub. Required if --push-to-hub is True.",
  ),
  push_to_hub: bool = typer.Option(
    False,
    "--push-to-hub",
    "-p",
    help="Push the processed dataset to Hugging Face Hub.",
  ),
  calculate_duration: bool = typer.Option(
    False,
    "--calc-duration",
    help="Calculate and print the total duration of audio files in hours.",
  ),
):
  """
  Processes an audio dataset, calculates total duration, and optionally pushes it to Hugging Face Hub.
  """
  data = []
  sum_duration = 0

  if push_to_hub:
    if commit_message is None:
      typer.echo(
        "[WARNING]: --commit should be provided when --push-to-hub is True. Using default commit message.",
        err=True,
      )
      commit_message = "No commit"

    if private:
      if commit_message is None:
        typer.echo(
          "[WARNING]: --private should be provided when --push-to-hub is True. Set private=True as a default.",
          err=True,
        )
        private = True

  typer.echo(f"Starting data processing from: {input_dir}")

  for speaker_dir in input_dir.iterdir():
    if not speaker_dir.is_dir():
      typer.echo(f"Skipping {speaker_dir}: not a directory.", err=True)
      continue

    speaker_id = speaker_dir.name
    typer.echo(f"Processing speaker: {speaker_id}")

    for wav_path in sorted(speaker_dir.glob("*.wav")):
      if not wav_path.exists() or not wav_path.is_file():
        typer.echo(
          f"[WARNING]: Skipping {wav_path}: not a file or does not exist.", err=True
        )
        continue

      txt_path = wav_path.with_suffix(".txt")
      if not txt_path.exists() or not txt_path.is_file():
        typer.echo(
          f"[WARNING]: Skipping {txt_path}: text file does not exist.", err=True
        )
        continue

      if calculate_duration:
        try:
          sum_duration += librosa.get_duration(filename=wav_path)
        except Exception as e:
          typer.echo(
            f"[ERROR]: Error calculating duration for {wav_path}: {e}", err=True
          )
          continue

      # Only append data if pushing to hub is requested
      if push_to_hub:
        with txt_path.open(
          "r", encoding="utf-8"
        ) as f:  # Specify encoding for text files
          text = f.read().strip()
        text = process_text(text)
        data.append(
          {"audio": str(wav_path.resolve()), "text": text, "speaker_id": speaker_id}
        )

  if calculate_duration:
    typer.echo(f"[INFO]: Total duration: {sum_duration / 3600:.2f} hours")

  if push_to_hub:
    if not data:
      typer.echo(
        "[ERROR]: No data collected to push. Ensure --input-dir is correct and contains files.",
        err=True,
      )
      raise typer.Exit(code=1)
    typer.echo(f"[INFO]: Collected {len(data)} entries for pushing to Hugging Face.")
    push_to_hf(data, dataset_id, commit=commit_message, private=private)


if __name__ == "__main__":
  app()
