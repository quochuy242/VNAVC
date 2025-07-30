import random
import re
import shutil
from pathlib import Path
from typing import Dict, List

import soundfile as sf
import torch
import typer
from jiwer import wer
from num2words import num2words
from rich.console import Console
from rich.progress import (
  BarColumn,
  Progress,
  TextColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from typing_extensions import Annotated

# Initialize Typer app
app = typer.Typer(
  name="whisper-local-qc",
  help="A CLI tool to perform quality control (QC) on local audio datasets, structured by speaker and book.",
)

# Initialize Rich console for pretty printing
console = Console()


def get_book_samples_from_folder(base_folder: Path) -> Dict[str, List[Dict[str, str]]]:
  """
  Scans the local folder structure to group audio/text pairs by book.
  Expected structure: base_folder/speaker_id/book_id_idx.(wav|txt)
  Returns a dictionary: {book_id: [(audio_path, text_path), ...]}
  """
  book_data = {}

  # Iterate through speaker folders
  for speaker_path in base_folder.iterdir():
    if not speaker_path.is_dir():
      continue

    # Iterate through .wav files in speaker folder
    for audio_file in speaker_path.glob("*.wav"):
      base_name = audio_file.stem  # Get file name without extension
      text_file = audio_file.with_suffix(".txt")

      if text_file.exists():
        # Extract book_id (e.g., "book_id_A_001" -> "book_id_A")
        parts = base_name.rsplit("_", 1)
        if len(parts) < 2 or not parts[1].isdigit():
          console.print(
            f"[yellow]Warning: Skipping {base_name}. Filename format does not match {{book_id}}_{{idx}}.[/yellow]"
          )
          continue
        book_id_prefix = parts[0]

        if book_id_prefix not in book_data:
          book_data[book_id_prefix] = []
        book_data[book_id_prefix].append(
          {
            "audio_path": audio_file,
            "text_path": text_file,
            "original_base_name": base_name,
            "speaker_id": speaker_path.name,
          }
        )
      else:
        console.print(
          f"[yellow]Warning: No corresponding .txt file found for {audio_file}. Remove it.[/yellow]"
        )
        audio_file.unlink()
  return book_data


def move_book_to_unqualified(book_samples: list, unqualified_folder: Path):
  """Moves all files belonging to a book to the unqualified_folder."""
  console.print(
    f"  [red]Moving {len(book_samples)} files ({book_samples[0]['speaker_id']} - {book_samples[0]['original_base_name']}) to {unqualified_folder}...[/red]"
  )
  unqualified_folder.mkdir(parents=True, exist_ok=True)

  for sample in book_samples:
    new_path = unqualified_folder / sample["speaker_id"]
    new_path.mkdir(parents=True, exist_ok=True)
    try:
      # Move .wav file
      shutil.move(sample["audio_path"], new_path / sample["audio_path"].name)
      # Move .txt file
      shutil.move(sample["text_path"], new_path / sample["text_path"].name)
    except Exception as e:
      console.print(
        f"[bold red]Error moving file {sample['original_base_name']}: {e}[/bold red]"
      )


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
def run_qc(
  local_dataset_path: Annotated[
    Path, typer.Option(help="Local path to the dataset folder (e.g., './dataset').")
  ] = Path("./dataset"),
  unqualified_output_path: Annotated[
    Path,
    typer.Option(
      help="Local path to move unqualified books (e.g., './unqualified_dataset')."
    ),
  ] = Path("./unqualified_dataset"),
  model_id: Annotated[
    str,
    typer.Option(
      help="Hugging Face model ID for the Whisper ASR model (e.g., 'vinai/PhoWhisper-large')."
    ),
  ] = "openai/whisper-small",
  sampling_percentage_per_book: Annotated[
    float,
    typer.Option(
      min=0.01,
      max=1.0,
      help="Percentage of samples to randomly select for QC *per book* (e.g., 0.1 for 10%).",
    ),
  ] = 0.05,
  wer_threshold: Annotated[
    float,
    typer.Option(
      min=0.0,
      max=100.0,
      help="Maximum acceptable Word Error Rate (WER) percentage for a book. If exceeded, the book is moved.",
    ),
  ] = 50.0,
  use_gpu: Annotated[
    bool,
    typer.Option(
      help="Use GPU if available. If false, forces CPU usage. (Note: Large models on CPU will be very slow)."
    ),
  ] = True,
):
  """
  Performs quality control on a local audio dataset, structured by speaker and book.
  Transcribes a sample of audio files per book using Whisper, compares against
  original text using WER. Moves entire books to an 'unqualified' folder if WER threshold is exceeded.
  """

  # --- Step 1: Load Model and Processor ---
  console.print(f"Loading Whisper model: {model_id}")

  if use_gpu and torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
    console.print("[green]Using GPU for inference.[/green]")
  else:
    device = "cpu"
    torch_dtype = torch.float32
    console.print(
      "[yellow]Using CPU for inference. This might be very slow for large models.[/yellow]"
    )

  processor = AutoProcessor.from_pretrained(model_id)
  model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
  ).to(device)

  # --- Step 2: Load Dataset from Local Folder and Group by Book ---
  local_dataset_path = Path(local_dataset_path)
  console.print(f"Scanning local dataset folder: {local_dataset_path}")
  if not local_dataset_path.is_dir():  # Use is_dir() for Path
    console.print(
      f"[bold red]Error: Local dataset path '{local_dataset_path}' does not exist or is not a directory.[/bold red]"
    )
    typer.Exit(code=1)

  book_data = get_book_samples_from_folder(local_dataset_path)
  if not book_data:
    console.print(
      "[yellow]No valid book data found in the specified local dataset path. Ensure correct file naming and structure.[/yellow]"
    )
    return  # Exit if no books found

  console.print(f"Found {len(book_data)} unique books to process.")

  # --- Step 3: Process Each Book with Progress Bar ---
  with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
    TimeElapsedColumn(),
    console=console,
    transient=False,  # Keep progress bar on screen after completion
  ) as progress:  # Renamed to 'progress' for clarity
    book_task = progress.add_task(
      "[green]Overall Book Processing[/green]", total=len(book_data)
    )

    for book_id, samples_in_book in book_data.items():
      console.print(
        f"\n[bold underline]--- Processing Book: {samples_in_book[0]['speaker_id']}-{book_id} (Total samples: {len(samples_in_book)}) ---[/bold underline]"
      )

      # Get random samples for this book
      num_samples_to_take = int(len(samples_in_book) * sampling_percentage_per_book)
      if num_samples_to_take == 0 and len(samples_in_book) > 0:
        num_samples_to_take = 1  # Take at least 1 sample if book has samples
      elif num_samples_to_take == 0:
        console.print("[yellow]  No samples to take for this book. Skipping.[/yellow]")
        progress.advance(book_task)
        continue

      sampled_book_samples = random.sample(samples_in_book, num_samples_to_take)
      console.print(
        f"  Randomly selected {len(sampled_book_samples)} samples ({sampling_percentage_per_book * 100:.0f}%) for WER calculation."
      )

      references = []  # Original text from local .txt file (Reference)
      hypotheses = []  # Text transcribed by Whisper (Hypothesis)

      # Create a new task for samples within the SAME progress instance
      sample_task = progress.add_task(
        f"[cyan]Transcribing Samples for '{book_id}'[/cyan]",
        total=len(sampled_book_samples),
      )

      for i, sample_info in enumerate(sampled_book_samples):
        # Read original text from local .txt file
        try:
          with open(sample_info["text_path"], "r", encoding="utf-8") as f:
            original_text = f.read().strip()
        except Exception as e:
          console.print(
            f"[bold red]Error reading text file {sample_info['text_path']}: {e}. Skipping sample.[/bold red]"
          )
          progress.advance(sample_task)
          continue

        # Load audio from local .wav file
        try:
          # Cố gắng đọc bằng soundfile trước
          data, sr = sf.read(sample_info["audio_path"])

          # Resample nếu cần
          if sr != 16000:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000

          audio_array = data
          audio_sampling_rate = sr

        except Exception as e:
          console.print(
            f"[bold red]Error reading audio file with soundfile: {sample_info['audio_path']}: {e}. Skipping sample.[/bold red]"
          )
          progress.advance(sample_task)
          continue

        # Preprocess audio with Whisper
        inputs = processor(
          audio_array, sampling_rate=audio_sampling_rate, return_tensors="pt"
        )
        input_features = inputs.input_features.to(device, torch_dtype)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
          attention_mask = attention_mask.to(device)

        # Generate transcription using Whisper
        with torch.no_grad():
          predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            language="vi",
            task="transcribe",
          )

        transcribed_text = processor.batch_decode(
          predicted_ids, skip_special_tokens=True
        )[0]

        # Normalize text for accurate WER calculation
        references.append(process_text(original_text.lower().strip()))
        hypotheses.append(transcribed_text.lower().strip())

        progress.update(
          sample_task,
          description=f"[cyan]Transcribing Sample: {sample_info['original_base_name']}[/cyan]",
        )
        progress.advance(sample_task)

      # Remove the sample task from the progress bar
      progress.remove_task(sample_task)

      # --- Step 4: Calculate WER for the Book and Make Decision ---
      if len(references) > 0:
        book_avg_wer = wer(references, hypotheses) * 100
        console.print(
          f"\n[bold green]--- Book '{book_id}' WER Results ---[/bold green]"
        )
        console.print(
          f"[bold green]Average WER for book '{book_id}':[/bold green] {book_avg_wer:.2f}%"
        )

        if (
          book_avg_wer >= wer_threshold
        ):  # Changed to >= as per "if not then move" logic
          console.print(
            f"\n[bold red]⚠️ Book '{book_id}' WER ({book_avg_wer:.2f}%) EXCEEDS allowed threshold ({wer_threshold}%).[/bold red]"
          )
          console.print(
            f"[bold red]ACTION: Moving all samples of book '{book_id}' to '{unqualified_output_path}' and deleting the book folder.[/bold red]"
          )
          move_book_to_unqualified(samples_in_book, unqualified_output_path)
        else:
          console.print(
            f"\n[bold green]✅ Book '{book_id}' WER ({book_avg_wer:.2f}%) is WITHIN the allowed threshold ({wer_threshold}%).[/bold green]"
          )
          console.print(
            f"[bold green]ACTION: Keeping all samples of book '{book_id}'.[/bold green]"
          )
      else:
        console.print(
          f"[yellow]  No samples were successfully processed for book '{book_id}' to calculate WER. Skipping decision for this book.[/yellow]"
        )

      # --- Memory Cleanup after each book ---
      console.print("[dim]  Attempting to clear memory...[/dim]")
      del references
      del hypotheses
      # Explicitly delete input features and predicted_ids if they're large
      # Check if variables exist before deleting to avoid NameError if skipped previously
      if "input_features" in locals():
        del input_features
      if "predicted_ids" in locals():
        del predicted_ids

      if device.startswith("cuda"):
        torch.cuda.empty_cache()
      import gc

      gc.collect()
      console.print("[dim]  Memory cleanup attempt completed.[/dim]")

      progress.advance(book_task)  # Advance overall progress bar

  console.print("\n[bold green]--- QC Process Completed ---[/bold green]")
  console.print(f"Qualified books remain in '{local_dataset_path}'.")
  console.print(f"Unqualified books have been moved to '{unqualified_output_path}'.")


if __name__ == "__main__":
  app()
