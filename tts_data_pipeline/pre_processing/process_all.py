#!/usr/bin/env python3
"""
Optimized TTS Data Processing Pipeline

Features:
- Modern libraries (pydantic, asyncio, pathlib)
- Configurable parallel processing
- Better error handling and logging
- Type hints throughout
- Modular architecture
- Progress tracking with rich
- Memory-efficient processing
- Retry mechanisms
"""

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Modern libraries
import aiofiles
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.progress import (
  Progress,
  SpinnerColumn,
  TextColumn,
  BarColumn,
  TimeRemainingColumn,
)
from rich.logging import RichHandler
from loguru import logger
import typer
from typing_extensions import Annotated

# Text processing
import pymupdf
import underthesea
import re
import string
from unicodedata import normalize

# Audio processing
import librosa
import soundfile as sf
import numpy as np

# TTS data pipeline imports
from tts_data_pipeline import constants

# Warning setup
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Setup rich console
console = Console()

# Configure logger with rich
logger.remove()
logger.add(
  RichHandler(console=console, rich_tracebacks=True),
  level="INFO",
  format=constants.FORMAT_LOG,
)


class ProcessingConfig(BaseModel):
  """Configuration for the processing pipeline"""

  # Directories
  text_pdf_dir: Path = Field(
    default=Path("data/pdfs"), description="Directory containing PDF files"
  )
  audio_raw_dir: Path = Field(
    default=Path("data/audio_raw"), description="Directory containing raw audio files"
  )
  text_output_dir: Path = Field(
    default=Path("data/text_processed"),
    description="Output directory for processed text",
  )
  audio_qualified_dir: Path = Field(
    default=Path("data/audio_qualified"),
    description="Output directory for qualified audio",
  )
  audio_unqualified_dir: Path = Field(
    default=Path("data/audio_unqualified"),
    description="Output directory for unqualified audio",
  )
  metadata_path: Path = Field(
    default=Path("data/metadata.csv"), description="Path to metadata file"
  )
  log_dir: Path = Field(default=Path("logs"), description="Directory for log files")

  # Processing parameters
  min_word_threshold: int = Field(
    default=20, ge=1, description="Minimum words per sentence group"
  )
  min_sample_rate: int = Field(
    default=16000, ge=8000, description="Minimum sample rate for audio files"
  )
  target_sample_rate: int = Field(
    default=22050, ge=8000, description="Target sample rate for output audio"
  )
  max_workers: int = Field(
    default=None, description="Maximum number of worker processes"
  )
  chunk_size: int = Field(default=100, ge=1, description="Processing chunk size")

  # Audio processing
  audio_format: str = Field(default="wav", description="Output audio format")
  audio_bitrate: str = Field(default="192k", description="Audio bitrate")
  normalize_audio: bool = Field(default=True, description="Normalize audio levels")

  # Text processing
  remove_punctuation: bool = Field(
    default=True, description="Remove punctuation from text"
  )
  normalize_unicode: bool = Field(default=True, description="Normalize unicode text")
  filter_empty_sentences: bool = Field(
    default=True, description="Filter out empty sentences"
  )

  # Flags
  update_metadata: bool = Field(default=True, description="Update metadata file")
  save_intermediate: bool = Field(
    default=False, description="Save intermediate processing files"
  )
  remove_original_files: bool = Field(
    default=False, description="Remove original files after processing"
  )

  @field_validator("max_workers")
  def validate_max_workers(cls, v):
    if v is None:
      return min(os.cpu_count(), 8)  # Cap at 8 for memory efficiency
    return min(v, os.cpu_count())

  class Config:
    arbitrary_types_allowed = True


@dataclass
class ProcessingResult:
  """Result of processing operation"""

  success: bool
  filename: str
  error: Optional[str] = None
  metadata: Dict[str, Any] = field(default_factory=dict)


class VietnameseSemioticNormalizer:
  """Modern Vietnamese text normalization with improved patterns"""

  def __init__(self):
    # Improved regex patterns
    self.patterns = {
      "number": re.compile(r"\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b"),
      "currency": re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:VND|vnđ|đ|USD|\$|€|¥|₫)\b", re.IGNORECASE
      ),
      "date": re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b"),
      "time": re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b"),
      "phone": re.compile(r"\b(?:\+84|0)\d{9,10}\b"),
      "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
      "url": re.compile(r"https?://[^\s]+"),
      "address": re.compile(r"\b(?:Số\s*\d+,?\s*)?(?:đường|Đường|phố|Phố)\s+[\w\s]+\b"),
    }

    # Translation table for cleaning
    self.punct_translator = str.maketrans("", "", string.punctuation)

  def normalize_text(self, text: str) -> str:
    """Comprehensive text normalization"""
    if not text:
      return ""

    # Remove URLs first
    text = self.patterns["url"].sub("", text)

    # Normalize unicode
    text = normalize("NFC", text)

    # Apply underthesea normalization
    try:
      text = underthesea.text_normalize(text)
    except Exception as e:
      logger.warning(f"Underthesea normalization failed: {e}")

    # Custom normalizations
    text = self._normalize_numbers(text)
    text = self._normalize_currency(text)
    text = self._normalize_dates(text)
    text = self._normalize_numbers_to_words(text)

    # Clean and standardize
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # Multiple spaces to single space
    text = text.lower()

    return text

  def _normalize_numbers(self, text: str) -> str:
    """Normalize number formats"""

    def replace_number(match):
      num_str = match.group()
      # Convert Vietnamese number format to standard
      if "," in num_str and "." in num_str:
        # Handle both comma and dot
        if num_str.index(",") < num_str.index("."):
          num_str = num_str.replace(",", "")
        else:
          num_str = num_str.replace(".", "").replace(",", ".")
      return num_str

    return self.patterns["number"].sub(replace_number, text)

  def _normalize_numbers_to_words(self, text: str) -> str:
    def replace_number(match):
      num_str = match.group()
      num_str = num_str.replace(",", "").replace(".", "")
      try:
        number = int(num_str)
        return self.number_to_vietnamese_words(number)
      except Exception as e:
        logger.error(f"Error converting number to words: {e}")
        return num_str

    return self.patterns["number"].sub(replace_number, text)

  def number_to_vietnamese_words(self, num: int) -> str:
    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    if num < 10:
      return digits[num]
    return " ".join(digits[int(d)] for d in str(num))

  def _normalize_currency(self, text: str) -> str:
    """Normalize currency representations"""

    def replace_currency(match):
      value = match.group()
      # Standardize currency symbols
      replacements = {"đ": "VND", "vnđ": "VND", "$": "USD", "₫": "VND"}
      for old, new in replacements.items():
        value = value.replace(old, new)
      return value.upper()

    return self.patterns["currency"].sub(replace_currency, text)

  def _normalize_dates(self, text: str) -> str:
    def replace_date(match):
      day, month, year = match.groups()
      if len(year) == 2:
        year = f"20{year}" if int(year) <= 30 else f"19{year}"

      day_words = self.number_to_vietnamese_words(int(day))
      month_words = self.number_to_vietnamese_words(int(month))
      year_words = " ".join([self.number_to_vietnamese_words(int(d)) for d in year])

      return f"ngày {day_words} tháng {month_words} năm {year_words}"

    return self.patterns["date"].sub(replace_date, text)


class TextProcessor:
  """Modern text processing with improved performance"""

  def __init__(self, config: ProcessingConfig):
    self.config = config
    self.normalizer = VietnameseSemioticNormalizer()

  async def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
    """Extract text from PDF using PyMuPDF (faster than pdfplumber)"""
    try:
      doc = pymupdf.open(str(pdf_path))
      text_parts = []

      for page in doc:
        text_parts.append(page.get_text())

      doc.close()
      return " ".join(text_parts)

    except Exception as e:
      logger.error(f"Error extracting text from {pdf_path}: {e}")
      return None

  def remove_punctuations(self, sentence: str):
    translator = str.maketrans("", "", string.punctuation)
    return sentence.translate(translator)

  def process_sentences(self, text: str) -> List[str]:
    """Process and group sentences efficiently"""
    if not text:
      return []

    # Tokenize into sentences
    sentences = underthesea.sent_tokenize(text)

    # Normalize sentences
    normalized_sentences = []
    for sentence in sentences:
      processed = self.normalizer.normalize_text(sentence)
      if self.config.remove_punctuation:
        processed = self.remove_punctuations(processed)
      if processed and len(processed.split()) >= 3:  # Minimum word check
        normalized_sentences.append(processed)

    # Group sentences by word count
    return self._group_sentences(normalized_sentences)

  def _group_sentences(self, sentences: List[str]) -> List[str]:
    """Group sentences to meet minimum word threshold"""
    if not sentences:
      return []

    grouped = []
    current_group = []
    current_word_count = 0

    for sentence in sentences:
      words = sentence.split()
      word_count = len(words)

      if current_word_count + word_count >= self.config.min_word_threshold:
        if current_group:
          current_group.append(sentence)
          grouped.append(" ".join(current_group))
          current_group = []
          current_word_count = 0
        else:
          # Single sentence meets threshold
          grouped.append(sentence)
      else:
        current_group.append(sentence)
        current_word_count += word_count

    # Add remaining sentences if they exist
    if current_group and current_word_count >= self.config.min_word_threshold // 2:
      grouped.append(" ".join(current_group))

    return grouped

  async def process_pdf(self, pdf_path: Path) -> ProcessingResult:
    """Process a single PDF file"""
    try:
      # Extract text
      text = await self.extract_text_from_pdf(pdf_path)
      if not text:
        return ProcessingResult(False, pdf_path.name, "Failed to extract text")

      # Process sentences
      sentences = self.process_sentences(text)
      if not sentences:
        return ProcessingResult(False, pdf_path.name, "No valid sentences found")

      # Save processed text
      output_path = self.config.text_output_dir / f"{pdf_path.stem}.txt"
      output_path.parent.mkdir(parents=True, exist_ok=True)

      async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        for sentence in sentences:
          await f.write(sentence + "\n")

      # Prepare metadata
      metadata = {
        "word_count": len(text.split()),
        "sentence_count": len(sentences),
        "text_size": output_path.stat().st_size,
      }

      # Remove original if requested
      if self.config.remove_original_files:
        pdf_path.unlink()

      return ProcessingResult(True, pdf_path.name, metadata=metadata)

    except Exception as e:
      logger.error(f"Error processing {pdf_path}: {e}")
      return ProcessingResult(False, pdf_path.name, str(e))


class AudioProcessor:
  """Modern audio processing with librosa and soundfile"""

  def __init__(self, config: ProcessingConfig):
    self.config = config

  def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
    """Load audio file with librosa"""
    try:
      # Load with librosa for better format support
      audio_data, sr = librosa.load(
        str(audio_path), sr=None, mono=True, dtype=np.float32
      )
      return audio_data, sr
    except Exception as e:
      logger.error(f"Error loading audio {audio_path}: {e}")
      raise

  def resample_audio(self, y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    if orig_sr == target_sr:
      return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

  def normalize_audio(self, y: np.ndarray) -> np.ndarray:
    """Normalize audio levels"""
    if self.config.normalize_audio:
      # Peak normalization
      peak = np.max(np.abs(y))
      if peak > 0:
        y = y / peak * 0.95  # Leave some headroom
    return y

  def combine_audio_files(self, audio_paths: List[Path], output_path: Path) -> bool:
    """Combine multiple audio files"""
    try:
      combined_audio = []
      target_sr = self.config.target_sample_rate

      for audio_path in audio_paths:
        audio_data, sr = self.load_audio(audio_path)

        # Resample if needed
        if sr != target_sr:
          audio_data = self.resample_audio(audio_data, sr, target_sr)

        # Normalize
        audio_data = self.normalize_audio(audio_data)

        combined_audio.append(audio_data)

      # Concatenate all audio
      final_audio = np.concatenate(combined_audio)

      # Save combined audio
      output_path.parent.mkdir(parents=True, exist_ok=True)
      sf.write(str(output_path), final_audio, target_sr)

      return True

    except Exception as e:
      logger.error(f"Error combining audio files: {e}")
      return False

  def get_audio_info(self, audio_path: Path) -> Dict[str, Any]:
    """Get audio file information"""
    try:
      audio_data, sr = self.load_audio(audio_path)
      duration = len(audio_data) / sr

      return {
        "sample_rate": sr,
        "duration": duration,
        "audio_size": audio_path.stat().st_size,
      }
    except Exception as e:
      logger.error(f"Error getting audio info for {audio_path}: {e}")
      return {}

  def process_audiobook(
    self, audio_paths: List[Path], book_name: str
  ) -> ProcessingResult:
    """Process a complete audiobook"""
    try:
      # Check if single file or multiple files
      if len(audio_paths) == 1:
        # Single file - just convert and check
        input_path = audio_paths[0]
        audio_data, sr = self.load_audio(input_path)

        # Check sample rate
        if sr < self.config.min_sample_rate:
          output_path = (
            self.config.audio_unqualified_dir
            / f"{book_name}.{self.config.audio_format}"
          )
          qualified = False
        else:
          output_path = (
            self.config.audio_qualified_dir / f"{book_name}.{self.config.audio_format}"
          )
          qualified = True

        # Resample if needed
        if sr != self.config.target_sample_rate:
          audio_data = self.resample_audio(
            audio_data, sr, self.config.target_sample_rate
          )
          sr = self.config.target_sample_rate

        # Normalize
        audio_data = self.normalize_audio(audio_data)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_data, sr)

      else:
        # Multiple files - combine them
        qualified_path = (
          self.config.audio_qualified_dir / f"{book_name}.{self.config.audio_format}"
        )
        unqualified_path = (
          self.config.audio_unqualified_dir / f"{book_name}.{self.config.audio_format}"
        )

        # Check sample rates of all files
        sample_rates = []
        for audio_path in audio_paths:
          _, sr = self.load_audio(audio_path)
          sample_rates.append(sr)

        min_sr = min(sample_rates)
        qualified = min_sr >= self.config.min_sample_rate
        output_path = qualified_path if qualified else unqualified_path

        # Combine files
        success = self.combine_audio_files(audio_paths, output_path)
        if not success:
          return ProcessingResult(False, book_name, "Failed to combine audio files")

      # Get final audio info
      audio_info = self.get_audio_info(output_path)

      # Remove original files if requested
      if self.config.remove_original_files:
        for audio_path in audio_paths:
          if audio_path.exists():
            audio_path.unlink()

      metadata = {
        "qualified": qualified,
        "sample_rate": audio_info.get("sample_rate", default=0),
        "duration": audio_info.get("duration", default=0),
        "audio_size": audio_info.get("audio_size", default=0),
      }

      return ProcessingResult(True, book_name, metadata=metadata)

    except Exception as e:
      logger.error(f"Error processing audiobook {book_name}: {e}")
      return ProcessingResult(False, book_name, str(e))


class AudiobookGrouper:
  """Groups audio files belonging to the same audiobook"""

  @staticmethod
  def group_audio_files(audio_dir: Path) -> Dict[str, List[Path]]:
    """Group audio files by audiobook name"""
    groups = {}

    # Support multiple audio formats
    extensions = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg"]

    # First, check if audio files are organized in subdirectories
    subdirs = [d for d in audio_dir.iterdir() if d.is_dir()]

    if subdirs:
      # Audio files are in subdirectories
      logger.info(f"Found {len(subdirs)} audiobook directories")

      for subdir in subdirs:
        book_name = subdir.name
        audio_files = []

        # Find all audio files in this subdirectory
        for ext in extensions:
          audio_files.extend(subdir.glob(ext))

        if audio_files:
          # Sort files by name (handle numbered sequences)
          audio_files.sort(
            key=lambda x: AudiobookGrouper._extract_sequence_number(x.name)
          )
          groups[book_name] = audio_files
          # logger.info(f"Found {len(audio_files)} audio files for '{book_name}'")

    else:
      # Audio files are directly in the main directory
      logger.info("Audio files found directly in main directory")
      audio_files = []

      # Find all audio files
      for ext in extensions:
        audio_files.extend(audio_dir.glob(ext))

      if not audio_files:
        logger.warning(f"No audio files found in {audio_dir}")
        return groups

      # Group by prefix (everything before first underscore or number)
      for file_path in audio_files:
        # Extract book name (everything before first underscore or number)
        name = file_path.stem

        # Try different splitting strategies
        book_name = name.split("_")[0]  # Split by underscore
        if book_name == name:
          # Try splitting by number
          match = re.match(r"([^\d]+)", name)
          if match:
            book_name = match.group(1).strip()

        if book_name not in groups:
          groups[book_name] = []
        groups[book_name].append(file_path)

      # Sort files within each group
      for book_name in groups:
        groups[book_name].sort(
          key=lambda x: AudiobookGrouper._extract_sequence_number(x.name)
        )

    return groups

  @staticmethod
  def _extract_sequence_number(filename: str) -> int:
    """Extract sequence number from filename for proper sorting"""
    # Look for number at the end of filename (before extension)
    match = re.search(r"(\d+)(?=\.\w+$)", filename)
    return int(match.group(1)) if match else 0


def process_audiobook_wrapper(args):
  book_name, paths, config = args
  try:
    processor = AudioProcessor(config)
    result = processor.process_audiobook(paths, book_name)
    return result
  except Exception as e:
    logger.error(f"Error processing audiobook {book_name}: {e}")
    return ProcessingResult(False, book_name, str(e))


class TTSProcessingPipeline:
  """Main pipeline class orchestrating the entire process"""

  def __init__(self, config: ProcessingConfig):
    self.config = config
    self.text_processor = TextProcessor(config)
    self.audio_processor = AudioProcessor(config)
    self.setup_directories()
    self.setup_logging()

  def setup_directories(self):
    """Create necessary directories"""
    directories = [
      self.config.text_output_dir,
      self.config.audio_qualified_dir,
      self.config.audio_unqualified_dir,
      self.config.log_dir,
    ]

    for directory in directories:
      directory.mkdir(parents=True, exist_ok=True)

  def setup_logging(self):
    """Setup file logging"""
    log_file = self.config.log_dir / "processing.log"
    logger.add(
      str(log_file),
      level="INFO",
      rotation="10 MB",
      encoding="utf-8",
      format=constants.FORMAT_LOG,
      colorize=False,
    )

  async def process_text_files(self) -> List[ProcessingResult]:
    """Process all PDF files with async concurrency"""
    pdf_files = list(self.config.text_pdf_dir.glob("*.pdf"))

    if not pdf_files:
      logger.warning(f"No PDF files found in {self.config.text_pdf_dir}")
      return []

    logger.info(f"Processing {len(pdf_files)} PDF files...")

    # Process with semaphore to control concurrency
    semaphore = asyncio.Semaphore(self.config.max_workers)

    async def process_with_semaphore(pdf_path):
      async with semaphore:
        return await self.text_processor.process_pdf(pdf_path)

    # Create tasks
    tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_files]

    # Process with progress tracking
    results = []
    with Progress(
      SpinnerColumn(),
      TextColumn("[progress.description]{task.description}"),
      BarColumn(),
      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
      TextColumn("[progress.completed]{task.completed}/{task.total}"),
      TimeRemainingColumn(),
      console=console,
    ) as progress:
      task = progress.add_task("Processing PDFs...", total=len(tasks))

      for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        progress.update(task, advance=1)

    return results

  def process_audio_files(self) -> List[ProcessingResult]:
    """Process all audio files with multiprocessing"""
    # Check if audio directory exists
    if not self.config.audio_raw_dir.exists():
      logger.error(f"Audio directory does not exist: {self.config.audio_raw_dir}")
      return []

    # Group audio files by audiobook
    audiobook_groups = AudiobookGrouper.group_audio_files(self.config.audio_raw_dir)

    if not audiobook_groups:
      logger.warning(f"No audio files found in {self.config.audio_raw_dir}")
      logger.info("Audio directory structure:")
      for item in self.config.audio_raw_dir.iterdir():
        logger.info(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
      return []

    logger.info(f"Processing {len(audiobook_groups)} audiobooks...")

    # # Log what we found
    # for book_name, paths in audiobook_groups.items():
    #   logger.info(f"Audiobook '{book_name}': {len(paths)} files")

    results = []
    tasks = [
      (book_name, paths, self.config) for book_name, paths in audiobook_groups.items()
    ]
    with ProcessPoolExecutor(
      max_workers=self.config.max_workers
    ) as executor:  # ProcessPoolExecutor, which use the function having module-level, for CPU-bound tasks
      future_to_book = {
        executor.submit(process_audiobook_wrapper, args): args[0]  # book_name
        for args in tasks
      }

      with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[progress.completed]{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
      ) as progress:
        task = progress.add_task("Processing Audio...", total=len(future_to_book))

        for future in as_completed(future_to_book):
          try:
            result = future.result()
            results.append(result)
            if result.success:
              logger.success(f"Successfully processed audiobook: {result.filename}")
            else:
              logger.error(
                f"Failed to process audiobook: {result.filename} - {result.error}"
              )
          except Exception as e:
            book_name = future_to_book[future]
            logger.error(f"Error processing {book_name}: {e}")
            results.append(ProcessingResult(False, book_name, str(e)))
          finally:
            progress.update(task, advance=1)

    return results

  def update_metadata(
    self, text_results: List[ProcessingResult], audio_results: List[ProcessingResult]
  ):
    """Update metadata file with processing results"""

    if not self.config.update_metadata:
      return

    # Load existing metadata or create new
    if self.config.metadata_path.exists():
      try:
        df = pd.read_csv(self.config.metadata_path)
      except Exception as e:
        logger.warning(
          f"Error loading metadata: {e}. Loading the after downloading metadata."
        )
        df = pd.read_csv(constants.AFTER_DOWNLOADING_METADATA_PATH)
        if df.empty:
          logger.warning("No existing metadata found, starting fresh.")
          df = pd.DataFrame()
    else:
      logger.info("Loading metadata from after downloading metadata path.")
      df = pd.read_csv(constants.AFTER_DOWNLOADING_METADATA_PATH)
      if df.empty:
        logger.warning("No existing metadata found, starting fresh.")
        df = pd.DataFrame()

    # Update with text results
    for result in text_results:
      if result.success:
        book_name = Path(result.filename).stem

        # Find or create row
        mask = (
          df["name"] == book_name
          if "name" in df.columns
          else pd.Series([False] * len(df))
        )

        if mask.any():
          # Update existing row
          idx = df[mask].index[0]
          for key, value in result.metadata.items():
            df.loc[idx, key] = value
        else:
          # Add new row
          new_row = {"name": book_name, **result.metadata}
          df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Update with audio results
    for result in audio_results:
      if result.success:
        book_name = result.filename

        # Find or create row
        mask = (
          df["name"] == book_name
          if "name" in df.columns
          else pd.Series([False] * len(df))
        )

        if mask.any():
          # Update existing row
          idx = df[mask].index[0]
          for key, value in result.metadata.items():
            df.loc[idx, key] = value
        else:
          # Add new row
          new_row = {"name": book_name, **result.metadata}
          df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save metadata
    self.config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(self.config.metadata_path, index=False)
    logger.info(f"Metadata updated: {self.config.metadata_path}")

  async def run(self):
    """Run the complete pipeline"""
    logger.info("Starting TTS Data Processing Pipeline...")

    # Process text files
    text_results = await self.process_text_files()
    successful_text = sum(1 for r in text_results if r.success)
    logger.info(
      f"Text processing completed: {successful_text}/{len(text_results)} files processed successfully"
    )
    if successful_text < len(text_results):  # Show errors if any
      for res in text_results:
        if not res.success:
          logger.error(f"Error processing text file {res.filename}: {res.error}")

    # Process audio files
    audio_results = self.process_audio_files()
    successful_audio = sum(1 for r in audio_results if r.success)
    logger.info(
      f"Audio processing completed: {successful_audio}/{len(audio_results)} files processed successfully"
    )
    if successful_audio < len(audio_results):  # Show errors if any
      for res in audio_results:
        if not res.success:
          logger.error(f"Error processing audio file {res.filename}: {res.error}")

    # Update metadata
    self.update_metadata(text_results, audio_results)

    # Summary
    console.print("\n[bold green]Pipeline Complete![/bold green]")
    console.print(f"✅ Text files processed: {successful_text}/{len(text_results)}")
    console.print(f"✅ Audio files processed: {successful_audio}/{len(audio_results)}")

    # Show errors if any
    errors = [r for r in text_results + audio_results if not r.success]
    if errors:
      console.print(f"\n[bold red]Errors encountered: {len(errors)}[/bold red]")
      for error in errors[:5]:  # Show first 5 errors
        console.print(f"❌ {error.filename}: {error.error}")
      if len(errors) > 5:
        console.print(f"... and {len(errors) - 5} more errors")


# CLI Interface
app = typer.Typer(help="TTS Data Processing Pipeline")


@app.command()
def process(
  config_file: Annotated[
    Optional[Path], typer.Option(help="Path to configuration file")
  ] = None,
  text_pdf_dir: Annotated[
    Optional[Path], typer.Option(help="Directory containing PDF files")
  ] = None,
  audio_raw_dir: Annotated[
    Optional[Path], typer.Option(help="Directory containing raw audio files")
  ] = None,
  max_workers: Annotated[
    Optional[int], typer.Option(help="Maximum number of worker processes")
  ] = None,
  min_sample_rate: Annotated[
    Optional[int], typer.Option(help="Minimum sample rate for audio")
  ] = None,
  remove_original: Annotated[
    bool, typer.Option(help="Remove original files after processing")
  ] = False,
):
  """Process TTS data with modern pipeline"""

  # Load configuration
  if config_file and config_file.exists():
    import json

    with open(config_file) as f:
      config_data = json.load(f)
    config = ProcessingConfig(**config_data)
  else:
    config = ProcessingConfig()

  # Override with CLI arguments
  if text_pdf_dir:
    config.text_pdf_dir = text_pdf_dir
  if audio_raw_dir:
    config.audio_raw_dir = audio_raw_dir
  if max_workers:
    config.max_workers = max_workers
  if min_sample_rate:
    config.min_sample_rate = min_sample_rate
  if remove_original is not None:
    config.remove_original_files = remove_original

  # Create and run pipeline
  pipeline = TTSProcessingPipeline(config)
  asyncio.run(pipeline.run())


@app.command()
def create_config(
  output_path: Annotated[
    Path, typer.Option(help="Output path for configuration file")
  ] = Path("config.json"),
):
  """Create a sample configuration file"""
  config = ProcessingConfig()

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


if __name__ == "__main__":
  app()
