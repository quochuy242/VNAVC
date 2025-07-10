import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import threading

import numpy as np
import pandas as pd
import typer
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
  BarColumn,
  Progress,
  SpinnerColumn,
  TextColumn,
  TimeRemainingColumn,
)
from typing_extensions import Annotated

from tts_data_pipeline import Book, Narrator, constants

# Configure logger
logger.remove()
logger.add(
  RichHandler(console=Console(), rich_tracebacks=True),
  level="INFO",
  format=constants.FORMAT_LOG,
)

# Setup console for rich logging
console = Console()

# Thread-safe semaphore for FFmpeg processes
ffmpeg_semaphore = threading.Semaphore(os.cpu_count())


@dataclass
class AlignmentResult:
  """Result of alignment process."""

  success: bool
  book_name: str
  alignment_path: Optional[str] = None
  error_message: Optional[str] = None
  segments_created: int = 0
  outliers_removed: int = 0


class AlignmentConfig(BaseModel):
  """Configuration for audio-text alignment."""

  # Directories
  audio_dir: Path = Field(
    default=Path("data/audio/processed"),
    description="Directory containing audio files",
  )
  text_dir: Path = Field(
    default=Path("data/text/processed"), description="Directory containing text files"
  )
  align_dir: Path = Field(
    default=Path("data/alignment"), description="Directory to save alignment results"
  )
  log_dir: Path = Field(default=Path("logs"), description="Directory to save log files")

  # Alignment parameters
  remove_outliers: bool = Field(
    default=True,
    description="Remove outlier segments based on duration",
  )
  remove_first: bool = Field(
    default=True,
    description="Remove first segment if it's too short",
  )
  split: bool = Field(
    default=False,
    description="Split audio and text files into segments based on alignment",
  )
  max_workers: int = Field(
    default=os.cpu_count(),
    description="Maximum number of workers to use for alignment",
  )

  @field_validator("audio_dir", "text_dir", "align_dir", "log_dir")
  def validate_paths(cls, v: Path) -> Path:
    """Ensure paths are absolute and exist."""
    if not v.exists():
      v.mkdir(parents=True, exist_ok=True)
    return v

  @field_validator("max_workers")
  def validate_max_workers(cls, v: int) -> int:
    """Ensure max_workers is a positive integer."""
    if v <= 0:
      console.print("[red]max_workers must be a positive integer[/red]")
      return 1
    if v > os.cpu_count():
      console.print(
        f"[yellow]max_workers ({v}) exceeds available CPU cores ({os.cpu_count()}). Using {os.cpu_count()} instead.[/yellow]"
      )
      return os.cpu_count()
    if v is None:
      v = min(os.cpu_count(), 8)
    logger.info(f"Using {v} workers for alignment")
    return v

  class Config:
    """Pydantic configuration."""

    arbitrary_types_allowed = True


class AudioTextAligner:
  """Audio-text alignment tool using aeneas."""

  def __init__(self, config: Optional[AlignmentConfig] = None):
    self.setup_logging()
    self.dependencies = ["ffmpeg", "ffprobe", "espeak"]
    self.check_dependencies()
    self.config = config or AlignmentConfig()
    self.setup_directories()

  def setup_logging(self):
    """Setup file logging."""
    log_file = self.config.log_dir / "alignment.log"
    logger.add(
      str(log_file),
      rotation="10 MB",
      encoding="utf-8",
      level="INFO",
      format=constants.FORMAT_LOG,
      colorize=False,
    )

  def setup_directories(self):
    """Ensure required directories exist."""
    directories = [
      self.config.audio_dir,
      self.config.text_dir,
      self.config.align_dir,
      self.config.log_dir,
    ]
    for directory in directories:
      directory.mkdir(parents=True, exist_ok=True)

  def check_dependencies(self) -> bool:
    """Check if required dependencies are installed."""
    missing = [dep for dep in self.dependencies if not shutil.which(dep)]
    if missing:
      logger.error(f"Missing dependencies: {', '.join(missing)}. Please install them.")
      return False
    return True

  def validate_files(self, audio_path: Path, text_path: Path) -> bool:
    """Validate that audio and text files exist and have correct extensions."""
    if not audio_path.exists():
      logger.error(f"Audio file not found: {audio_path}")
      return False

    if not text_path.exists():
      logger.error(f"Text file not found: {text_path}")
      return False

    # Check file extensions
    if audio_path.suffix.lower() != ".wav":
      logger.error(f"Unsupported audio format: {audio_path.suffix}")
      return False

    if text_path.suffix.lower() != ".txt":
      logger.error(f"Unsupported text format: {text_path.suffix}")
      return False

    return True

  def find_json_metadata(self, audio_path: Path) -> Optional[Path]:
    """Find the JSON metadata file for an audio file."""
    json_name = audio_path.name.replace(".wav", ".json")
    json_path = Path(constants.METADATA_SAVE_PATH) / json_name

    if json_path.exists():
      return json_path
    else:
      logger.warning(f"Metadata file not found: {json_path}")
      return None

  def get_output_directory(self, book: Book) -> Path:
    """Get the alignment output directory for a book."""
    if isinstance(book.narrator, Narrator):
      dir_name = str(book.narrator.id) if book.narrator.id else book.narrator.name
    elif isinstance(book.narrator, list):
      dir_name = (
        str(book.narrator[0].id) if book.narrator[0].id else book.narrator[0].name
      )
    else:
      dir_name = "Unknown"

    output_dir = Path(constants.DATASET_DIR) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

  def process_alignment_output(self, book: Book) -> List[int]:
    """Process alignment output and identify outliers."""
    # Load alignment data
    align_df = pd.read_csv(book.alignment_path, sep="\t", names=["start", "end", "id"])

    # Calculate duration and clean ID column
    align_df["duration"] = align_df["end"] - align_df["start"]
    align_df["id"] = align_df["id"].str.replace("f", "").astype(int)

    # Save processed alignment data
    align_df.to_csv(book.alignment_path, sep="\t", index=False, header=False)

    # Find outliers
    outlier_indices = []
    if self.config.remove_first:
      outlier_indices.append(0)

    if len(align_df) > 3:
      z_scores = np.abs(
        (align_df["duration"] - align_df["duration"].mean())
        / align_df["duration"].std()
      )
      outlier_mask = z_scores > 3.0
      outlier_indices.extend(np.where(outlier_mask)[0].tolist())

    # Save outlier indices
    outlier_indices = sorted(set(outlier_indices))
    outlier_file = Path(book.alignment_path).parent / "outlier.txt"
    with open(outlier_file, "w") as f:
      f.write("\n".join(str(idx) for idx in outlier_indices))

    logger.info(f"Found {len(outlier_indices)} outliers in alignment for {book.name}")
    return outlier_indices

  def _run_ffmpeg_sync(self, cmd: List[str]) -> None:
    """Run ffmpeg command synchronously with semaphore."""
    with ffmpeg_semaphore:
      try:
        subprocess.run(
          cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, text=True
        )
      except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.strip()}")
        raise

  def _create_audio_segment(self, args: Tuple[str, str, float, float, str]) -> bool:
    """Create a single audio segment using FFmpeg."""
    input_file, output_file, start_time, duration, book_name = args

    cmd = [
      "ffmpeg",
      "-y",  # Overwrite existing files
      "-i",
      input_file,
      "-ss",
      str(start_time),
      "-t",
      str(duration),
      "-c:a",
      "pcm_s16le",
      "-avoid_negative_ts",
      "make_zero",
      output_file,
    ]

    try:
      self._run_ffmpeg_sync(cmd)
      return True
    except subprocess.CalledProcessError:
      logger.error(f"Failed to create segment: {output_file}")
      return False

  def _create_text_segment(self, args: Tuple[str, int, str, str]) -> bool:
    """Create a single text segment."""
    output_file, line_id, text_content, book_name = args

    try:
      with open(output_file, "w", encoding="utf-8") as f:
        f.write(text_content)
      return True
    except Exception as e:
      logger.error(f"Failed to create text segment {output_file}: {e}")
      return False

  def remove_outlier_segments(self, book: Book, segments_created: int) -> int:
    """Remove outlier segments based on alignment data."""
    outlier_file = Path(book.alignment_path).parent / "outlier.txt"
    if not outlier_file.exists():
      return segments_created

    with open(outlier_file, "r", encoding="utf-8") as f:
      outlier_indices = [line.strip() for line in f if line.strip()]

    output_dir = self.get_output_directory(book)
    removed_count = 0

    for idx in outlier_indices:
      audio_file = output_dir / f"{book.id}_{idx}.wav"
      text_file = output_dir / f"{book.id}_{idx}.txt"

      if audio_file.exists():
        audio_file.unlink()
        removed_count += 1
      if text_file.exists():
        text_file.unlink()

    return segments_created - removed_count

  def split_audio_segments(self, book: Book) -> int:
    """Split audio file into segments based on alignment data using parallel processing."""
    if not book.audio_path:
      logger.error("Audio path is None")
      return 0

    output_dir = self.get_output_directory(book)
    align_df = pd.read_csv(
      book.alignment_path, sep="\t", names=["start", "end", "id", "duration"]
    )

    # Prepare arguments for parallel processing
    tasks = []
    for _, row in align_df.iterrows():
      line_id = int(row["id"])
      start_time = row["start"]
      duration = row["duration"]

      output_file = str(output_dir / f"{book.id}_{line_id}.wav")

      tasks.append((str(book.audio_path), output_file, start_time, duration, book.name))

    # Process segments in parallel
    segments_created = 0
    with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
      future_to_task = {
        executor.submit(self._create_audio_segment, task): task for task in tasks
      }

      for future in as_completed(future_to_task):
        if future.result():
          segments_created += 1

    logger.info(f"Created {segments_created} audio segments for {book.name}")

    # Remove outlier segments
    if self.config.remove_outliers:
      segments_created = self.remove_outlier_segments(book, segments_created)

    return segments_created

  def split_text_segments(self, book: Book) -> int:
    """Split text file into segments based on alignment data using parallel processing."""
    if not book.text_path:
      logger.error("Text path is None")
      return 0

    output_dir = self.get_output_directory(book)
    align_df = pd.read_csv(
      book.alignment_path, sep="\t", names=["start", "end", "id", "duration"]
    )

    # Read text file once
    with open(book.text_path, "r", encoding="utf-8") as f:
      text_lines = f.read().splitlines()

    # Prepare arguments for parallel processing
    tasks = []
    for _, row in align_df.iterrows():
      line_id = int(row["id"])

      if line_id >= len(text_lines):
        logger.warning(
          f"Line index {line_id} out of range for {book.text_path}, skipping segment."
        )
        continue

      output_file = str(output_dir / f"{book.id}_{line_id}.txt")
      text_content = text_lines[line_id]

      tasks.append((output_file, line_id, text_content, book.name))

    # Process segments in parallel
    segments_created = 0
    with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
      future_to_task = {
        executor.submit(self._create_text_segment, task): task for task in tasks
      }

      for future in as_completed(future_to_task):
        if future.result():
          segments_created += 1

    logger.info(f"Created {segments_created} text segments for {book.name}")
    return segments_created

  def align_book(self, book: Book) -> AlignmentResult:
    """Align a single book's audio and text files."""
    # Setup output directory
    os.makedirs(constants.AENEAS_OUTPUT_DIR, exist_ok=True)

    # Determine output path
    if not book.alignment_path:
      alignment_dir = Path(constants.AENEAS_OUTPUT_DIR) / book.name
      alignment_dir.mkdir(parents=True, exist_ok=True)
      output_path = alignment_dir / "output.tsv"
    else:
      output_path = Path(book.alignment_path)

    try:
      # Run aeneas alignment if not already done
      if not book.alignment_path or not Path(book.alignment_path).exists():
        logger.info(f"Starting alignment for {book.name}")

        task = Task(config_string=constants.AENEAS_CONFIG)
        task.audio_file_path_absolute = (
          str(book.audio_path.absolute()) if book.audio_path else None
        )
        task.text_file_path_absolute = (
          str(book.text_path.absolute()) if book.text_path else None
        )
        task.sync_map_file_path_absolute = str(output_path.absolute())

        ExecuteTask(task).execute()
        task.output_sync_map_file()

        book.update_paths(alignment_path=str(output_path))
        logger.info(f"Alignment completed for {book.name}")
      else:
        logger.info(f"Using existing alignment for {book.name}: {book.alignment_path}")

      # Process alignment output
      outlier_indices = self.process_alignment_output(book)

      segments_created = 0
      if self.config.split:
        # Split files with parallel processing
        audio_segments = self.split_audio_segments(book)
        text_segments = self.split_text_segments(book)
        segments_created = min(audio_segments, text_segments)

      return AlignmentResult(
        success=True,
        book_name=book.name,
        alignment_path=str(output_path),
        segments_created=segments_created,
        outliers_removed=len(outlier_indices),
      )

    except Exception as e:
      logger.exception(f"Alignment failed for {book.name}: {e}")
      return AlignmentResult(success=False, book_name=book.name, error_message=str(e))


# Typer CLI interface
app = typer.Typer(help="Audio-text alignment tool using aeneas")


@app.command()
def create_config(
  output_path: Annotated[
    Path, typer.Option(help="Output path for configuration file")
  ] = Path("config.json"),
):
  """Create a sample configuration file"""
  config = AlignmentConfig()

  # Convert to dict and save
  config_dict = config.model_dump()

  # Convert Path objects to strings for JSON serialization
  for key, value in config_dict.items():
    if isinstance(value, Path):
      config_dict[key] = str(value)

  import json

  with open(output_path, "w") as f:
    json.dump(config_dict, f, indent=2)

  console.print(f"Configuration file created: {output_path}")
  console.print("Edit this file to customize your processing parameters.")


@app.command()
def test(
  audio: str = typer.Option(..., "-a", "--audio", help="Path to the audio file"),
  text: str = typer.Option(..., "-t", "--text", help="Path to the text file"),
  split: bool = typer.Option(False, "-s", "--split", help="Split files into segments"),
  force: bool = typer.Option(
    False, "-f", "--force", help="Force realignment even if alignment exists"
  ),
  remove_first: bool = typer.Option(
    True, "--remove-first/--keep-first", help="Remove first segment (usually silence)"
  ),
  max_workers: int = typer.Option(
    os.cpu_count(), "--max-workers", help="Maximum number of workers for splitting"
  ),
):
  """Align audio and text files using aeneas."""
  config = AlignmentConfig(
    split=split, remove_first=remove_first, max_workers=max_workers
  )
  aligner = AudioTextAligner(config)

  # Check dependencies
  if not aligner.check_dependencies():
    typer.echo("Missing dependencies. Please install them and try again.")
    raise typer.Exit(1)

  # Validate input files
  audio_path = Path(audio)
  text_path = Path(text)

  if not aligner.validate_files(audio_path, text_path):
    typer.echo("File validation failed.")
    raise typer.Exit(1)

  # Setup book object
  json_path = aligner.find_json_metadata(audio_path)
  if json_path:
    book = Book.from_json(json_path)
    book.update_paths(text_path=text_path, audio_path=audio_path)
  else:
    typer.echo(f"Warning: No metadata found for {audio_path}")
    # Create a basic book object if no metadata is found
    book = Book(name=audio_path.stem, audio_path=audio_path, text_path=text_path)

  # Check for existing alignment
  existing_alignment = Path(constants.AENEAS_OUTPUT_DIR) / book.name / "output.tsv"
  if existing_alignment.exists() and not force:
    book.update_paths(alignment_path=str(existing_alignment))
    typer.echo(f"Using existing alignment: {existing_alignment}")
  elif force:
    typer.echo("Forcing realignment...")

  # Run alignment
  typer.echo(f"Processing book: {book.name}")
  result = aligner.align_book(book)

  # Report results
  if result.success:
    typer.echo(f"✅ Alignment successful for {result.book_name}")
    typer.echo(f"   Alignment saved to: {result.alignment_path}")
    if result.segments_created > 0:
      typer.echo(f"   Created {result.segments_created} segments")
    if result.outliers_removed > 0:
      typer.echo(f"   Removed {result.outliers_removed} outliers")
  else:
    typer.echo(f"❌ Alignment failed for {result.book_name}: {result.error_message}")
    raise typer.Exit(1)


@app.command()
def run(
  config_file: Annotated[
    Path, typer.Option("-c", "--config", help="Path to configuration file")
  ],
  audio_dir: Annotated[
    str,
    typer.Option("-d", "--audio-dir", help="Directory containing audio files"),
  ] = constants.AUDIO_PROCESSED_DIR,
  text_dir: Annotated[
    str,
    typer.Option("-t", "--text-dir", help="Directory containing text files"),
  ] = constants.TEXT_PROCESSED_DIR,
  split: Annotated[
    bool, typer.Option("-s", "--split/--no-split", help="Split files into segments")
  ] = False,
  max_workers: Annotated[
    int, typer.Option("--max-workers", help="Maximum number of workers for splitting")
  ] = os.cpu_count(),
):
  """Batch align multiple audio-text pairs."""
  if config_file and config_file.exists():
    import json

    with open(config_file, "r") as f:
      config_data = json.load(f)
    config = AlignmentConfig(**config_data)
  else:
    config = AlignmentConfig()

  # Override config with command line options
  if audio_dir:
    config.audio_dir = Path(audio_dir)
  if text_dir:
    config.text_dir = Path(text_dir)
  if split is not None:
    config.split = split
  if max_workers:
    config.max_workers = max_workers

  aligner = AudioTextAligner(config)

  audio_dir = Path(audio_dir)
  text_dir = Path(text_dir)
  if not audio_dir.exists() or not text_dir.exists():
    typer.echo(f"Audio directory: {audio_dir} or text directory: {text_dir} not found")
    raise typer.Exit(1)

  # Find audio files
  audio_files = sorted(audio_dir.glob("*.wav"))
  text_files = sorted(text_dir.glob("*.txt"))

  # Get the pairs of audio and text files
  audio_text_pairs = []
  idx_audio, idx_text = 0, 0
  while idx_audio < len(audio_files) and idx_text < len(text_files):
    audio_file = audio_files[idx_audio]
    text_file = text_files[idx_text]

    # Match audio and text files by name (ignoring extensions)
    if audio_file.stem == text_file.stem:
      audio_text_pairs.append((audio_file, text_file))
      idx_audio += 1
      idx_text += 1
    elif audio_file.stem < text_file.stem:
      idx_audio += 1
    else:
      idx_text += 1

  console.print(f"Found {len(audio_text_pairs)} audio-text pairs to align")
  console.print(f"Using {max_workers} workers for parallel processing")

  with Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]Aligning..."),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("[progress.completed]{task.completed}/{task.total}"),
    TimeRemainingColumn(),
    console=console,
  ) as progress:
    task = progress.add_task("Alignment", total=len(audio_text_pairs))
    successful = 0
    failed = 0
    for audio_file, text_file in audio_text_pairs:
      # Validate files
      if not aligner.validate_files(audio_file, text_file):
        console.print(
          f"❌ Validation failed for {audio_file.name} and {text_file.name}"
        )
        failed += 1
        continue

      # Print progress
      console.print(f"Aligning: {audio_file.name}")

      # Setup book
      json_path = aligner.find_json_metadata(audio_file)
      if json_path:
        book = Book.from_json(json_path)
        book.update_paths(text_path=text_file, audio_path=audio_file)
      else:
        book = Book(name=audio_file.stem, audio_path=audio_file, text_path=text_file)

      # Run alignment
      result = aligner.align_book(book)

      if result.success:
        successful += 1
        console.print("✅ Success")
      else:
        failed += 1
        console.print(f"❌ Failed: {result.error_message}")

      progress.update(
        task, advance=1, completed=successful + failed, total=len(audio_text_pairs)
      )

  console.print(
    f"\nBatch alignment completed: {successful} successful, {failed} failed"
  )


if __name__ == "__main__":
  app()
