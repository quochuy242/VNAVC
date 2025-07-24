# TTS Data Processing Pipeline

A comprehensive, high-performance data preprocessing pipeline for Text-to-Speech (TTS) systems, specifically designed for Vietnamese audiobooks. This pipeline handles both text extraction from PDFs and audio processing with advanced normalization and quality control.

## Overview

This pipeline transforms raw audiobook data into TTS-ready formats by:

- **Text Processing**: Extracting and normalizing Vietnamese text from PDF files
- **Audio Processing**: Converting, resampling, and quality-checking audio files
- **Metadata Management**: Tracking processing results and file statistics
- **Quality Control**: Separating qualified and unqualified audio based on sample rates

## Features

### 🚀 **High-Performance Processing**

- **Async Text Processing**: Concurrent PDF extraction with semaphore-controlled parallelism
- **Multiprocess Audio**: CPU-intensive audio processing with ProcessPoolExecutor
- **Memory Efficient**: Streaming audio processing with optimized memory usage
- **Progress Tracking**: Rich console interface with real-time progress bars

### 📝 **Advanced Text Processing**

- **PDF Extraction**: Fast text extraction using PyMuPDF
- **Vietnamese NLP**: Integration with `underthesea` for Vietnamese text processing
- **Semiotic Normalization**: Comprehensive text normalization (numbers, dates, currency)
- **Sentence Grouping**: Intelligent sentence grouping based on word count thresholds

### 🎵 **Professional Audio Processing**

- **Multi-format Support**: MP3, WAV, M4A, FLAC, OGG audio processing
- **Quality Control**: Automatic separation based on sample rate requirements
- **Audio Combining**: Merge multiple audio files per audiobook
- **Normalization**: Peak audio normalization with configurable parameters
- **Resampling**: High-quality resampling using librosa

### 🔧 **Flexible Configuration**

- **Pydantic Models**: Type-safe configuration with validation
- **CLI Interface**: Typer-based command-line interface
- **JSON Config**: External configuration file support
- **Directory Structure**: Configurable input/output directories

## Installation

### Prerequisites

```bash
# Core dependencies
pip install pandas pydantic typer rich loguru aiofiles

# Text processing
pip install pymupdf underthesea

# Audio processing
pip install librosa soundfile numpy

# Optional: For better performance
pip install numba  # Accelerates librosa operations
```

### System Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended for large audiobooks
- **Storage**: Adequate space for processed files (typically 2-3x original size)
- **CPU**: Multi-core recommended for audio processing

## Quick Start

### Basic Usage

```bash
# Process with default settings
python processing_pipeline.py process

# Specify custom directories
python processing_pipeline.py process \
  --text-pdf-dir ./pdfs \
  --audio-raw-dir ./audio_raw \
  --max-workers 4

# Remove original files after processing
python processing_pipeline.py process --remove-original
```

### Create Configuration File

```bash
# Generate default configuration
python processing_pipeline.py create-config --output-path my_config.json

# Use custom configuration
python processing_pipeline.py process --config-file my_config.json
```

## Configuration

### Default Configuration Structure

```json
{
  "text_pdf_dir": "data/pdfs",
  "audio_raw_dir": "data/audio_raw",
  "text_output_dir": "data/text_processed",
  "audio_qualified_dir": "data/audio_qualified",
  "audio_unqualified_dir": "data/audio_unqualified",
  "metadata_path": "data/metadata.csv",
  "log_dir": "logs",
  "min_word_threshold": 20,
  "min_sample_rate": 16000,
  "target_sample_rate": 22050,
  "max_workers": 8,
  "chunk_size": 100,
  "audio_format": "wav",
  "normalize_audio": true,
  "remove_punctuation": true,
  "normalize_unicode": true,
  "update_metadata": true,
  "remove_original_files": false
}
```

### Key Parameters

#### Text Processing

- **`min_word_threshold`**: Minimum words per sentence group (default: 20)
- **`remove_punctuation`**: Strip punctuation from processed text
- **`normalize_unicode`**: Apply Unicode normalization (NFC)
- **`filter_empty_sentences`**: Remove empty or too-short sentences

#### Audio Processing

- **`min_sample_rate`**: Minimum acceptable sample rate (16kHz default)
- **`target_sample_rate`**: Output sample rate for processed audio (22kHz default)
- **`normalize_audio`**: Apply peak normalization to audio
- **`audio_format`**: Output format (wav, mp3, flac)

#### Performance

- **`max_workers`**: Maximum parallel processes (auto-detected, capped at 8)
- **`chunk_size`**: Processing batch size

## Directory Structure

### Expected Input Structure

```
project/
├── data/
│   ├── pdfs/                 # Input PDF files
│   │   ├── book1.pdf
│   │   ├── book2.pdf
│   │   └── ...
│   └── audio_raw/            # Input audio files
│       ├── book1/            # Option 1: Organized in subdirectories
│       │   ├── book1_1.mp3
│       │   ├── book1_2.mp3
│       │   └── ...
│       ├── book2_1.mp3       # Option 2: Flat structure with prefixes
│       ├── book2_2.mp3
│       └── ...
```

### Generated Output Structure

```
project/
├── data/
│   ├── text_processed/       # Processed text files
│   │   ├── book1.txt
│   │   ├── book2.txt
│   │   └── ...
│   ├── audio_qualified/      # High-quality audio (≥16kHz)
│   │   ├── book1.wav
│   │   ├── book2.wav
│   │   └── ...
│   ├── audio_unqualified/    # Low-quality audio (<16kHz)
│   │   └── low_quality_book.wav
│   └── metadata.csv          # Processing results
└── logs/
    └── processing.log        # Detailed processing logs
```

## Text Processing Details

### Vietnamese Text Normalization

The pipeline includes sophisticated Vietnamese text processing:

#### Semiotic Normalization

- **Numbers**: Convert to Vietnamese words (123 → "một hai ba")
- **Currency**: Standardize currency symbols (đ → VND)
- **Dates**: Convert to spoken format ("12/01/2023" → "ngày mười hai tháng một năm hai nghìn không hai mười ba")
- **Phone Numbers**: Normalize Vietnamese phone formats
- **URLs/Emails**: Remove or standardize web content

#### Text Cleaning

```python
# Example normalization
input_text = "Giá sách là 150,000đ vào ngày 15/3/2023"
output_text = "giá sách là một trăm năm mười nghìn vnd vào ngày mười lăm tháng ba năm hai nghìn không hai mười ba"
```

#### Sentence Grouping

- Groups sentences to meet minimum word thresholds
- Preserves semantic boundaries
- Configurable grouping parameters

### PDF Text Extraction

```python
# High-performance PDF processing
async def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = pymupdf.open(str(pdf_path))
    text_parts = [page.get_text() for page in doc]
    return " ".join(text_parts)
```

## Audio Processing Details

### Multi-file Audiobook Handling

The pipeline automatically detects and handles different audiobook organizations:

1. **Subdirectory Structure**: `book_name/part1.mp3, book_name/part2.mp3`
2. **Prefix Structure**: `book_name_1.mp3, book_name_2.mp3`
3. **Single Files**: `book_name.mp3`

### Audio Quality Control

```python
# Quality assessment
if sample_rate >= min_sample_rate:
    # High quality: Move to qualified directory
    output_dir = audio_qualified_dir
else:
    # Low quality: Move to unqualified directory
    output_dir = audio_unqualified_dir
```

### Audio Processing Pipeline

1. **Load**: Multi-format audio loading with librosa
2. **Combine**: Concatenate multiple parts if needed
3. **Resample**: Convert to target sample rate
4. **Normalize**: Peak normalization with headroom
5. **Save**: Export in specified format

## API Reference

### Core Classes

#### ProcessingConfig

```python
from processing_pipeline import ProcessingConfig

config = ProcessingConfig(
    text_pdf_dir=Path("my_pdfs/"),
    audio_raw_dir=Path("my_audio/"),
    min_sample_rate=22050,
    max_workers=6
)
```

#### TTSProcessingPipeline

```python
from processing_pipeline import TTSProcessingPipeline

pipeline = TTSProcessingPipeline(config)
await pipeline.run()
```

### Individual Processors

#### TextProcessor

```python
from processing_pipeline import TextProcessor

processor = TextProcessor(config)
result = await processor.process_pdf(pdf_path)
```

#### AudioProcessor

```python
from processing_pipeline import AudioProcessor

processor = AudioProcessor(config)
result = processor.process_audiobook(audio_paths, book_name)
```

### Vietnamese Text Normalizer

```python
from processing_pipeline import VietnameseSemioticNormalizer

normalizer = VietnameseSemioticNormalizer()
normalized = normalizer.normalize_text("Có 123 quyển sách")
# Output: "có một hai ba quyển sách"
```

## Metadata Output

The pipeline generates comprehensive metadata in CSV format:

```csv
name,author,duration,word_count,sentence_count,sample_rate,qualified,audio_size,text_size
book1,Author Name,3600.5,15000,750,22050,true,45000000,125000
book2,Another Author,2400.2,12000,600,8000,false,20000000,98000
```

### Metadata Fields

- **Basic Info**: name, author, duration
- **Text Metrics**: word_count, sentence_count, text_size
- **Audio Metrics**: sample_rate, audio_size, qualified status
- **Processing**: timestamps, success/failure status

## Performance Optimization

### Memory Management

- **Streaming Processing**: Large files processed in chunks
- **Garbage Collection**: Explicit cleanup of audio arrays
- **Connection Pooling**: Efficient resource usage

### Parallelism Strategy

- **Text**: Async I/O for PDF extraction and file writing
- **Audio**: Process-based parallelism for CPU-intensive operations
- **Semaphore Control**: Prevents resource exhaustion

### Benchmarks

- **Text Processing**: ~50 PDFs/minute (varies by size)
- **Audio Processing**: ~10 audiobooks/minute (depends on length and CPU)
- **Memory Usage**: ~2-4GB for typical workloads

## Error Handling

### Robust Error Recovery

```python
# Comprehensive error handling
try:
    result = await processor.process_pdf(pdf_path)
except Exception as e:
    logger.error(f"PDF processing failed: {e}")
    return ProcessingResult(False, pdf_path.name, str(e))
```

### Common Issues and Solutions

**PDF Extraction Failures**

- **Cause**: Corrupted or encrypted PDFs
- **Solution**: File validation and fallback extraction methods

**Audio Processing Errors**

- **Cause**: Unsupported formats or corrupted files
- **Solution**: Format detection and graceful degradation

**Memory Issues**

- **Cause**: Large audio files or too many parallel processes
- **Solution**: Reduce `max_workers` or enable chunked processing

## Monitoring and Logging

### Rich Console Interface

- Real-time progress bars
- Color-coded status messages
- Summary statistics

### Structured Logging

```python
# Comprehensive logging
logger.info("Processing started")
logger.success("File processed successfully")
logger.error("Processing failed", extra={"file": filename, "error": str(e)})
```

### Log Files

- **Location**: `logs/processing.log`
- **Rotation**: 10MB file rotation
- **Format**: Structured with timestamps and context

## Advanced Usage

### Custom Text Normalization

```python
class CustomNormalizer(VietnameseSemioticNormalizer):
    def custom_normalize(self, text: str) -> str:
        # Add custom normalization rules
        text = super().normalize_text(text)
        # Your custom processing
        return text
```

### Batch Processing Script

```python
import asyncio
from pathlib import Path

async def batch_process():
    configs = [
        ProcessingConfig(text_pdf_dir=Path(f"batch_{i}"))
        for i in range(5)
    ]

    pipelines = [TTSProcessingPipeline(config) for config in configs]
    await asyncio.gather(*[pipeline.run() for pipeline in pipelines])
```

### Integration with Training Pipelines

```python
# Export for TTS training
def prepare_training_data(metadata_path: Path):
    df = pd.read_csv(metadata_path)
    qualified_books = df[df['qualified'] == True]

    # Generate training manifest
    manifest = []
    for _, row in qualified_books.iterrows():
        manifest.append({
            "audio_filepath": f"audio_qualified/{row['name']}.wav",
            "text_filepath": f"text_processed/{row['name']}.txt",
            "duration": row['duration']
        })

    return manifest
```

## Troubleshooting

### Performance Issues

- **Slow PDF Processing**: Check PDF complexity and reduce `max_workers`
- **High Memory Usage**: Enable chunked processing for large files
- **Audio Quality Issues**: Verify input sample rates and formats

### File Organization

- **Missing Files**: Ensure proper directory structure
- **Naming Conflicts**: Use consistent naming conventions
- **Permission Errors**: Check file system permissions

### Dependencies

```bash
# Update dependencies
pip install --upgrade librosa soundfile pymupdf underthesea

# Install optional accelerations
pip install numba  # For librosa acceleration
conda install -c conda-forge ffmpeg  # For additional audio format support
```

## Contributing

1. **Code Style**: Follow PEP 8 and use type hints
2. **Testing**: Add tests for new normalization rules
3. **Documentation**: Update docstrings and examples
4. **Performance**: Profile changes for performance impact

## License

This processing pipeline is part of the TTS Data Pipeline project. Please refer to the main project license for usage terms.
