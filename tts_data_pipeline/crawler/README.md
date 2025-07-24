# TTS Data Pipeline - Crawler Module

A comprehensive audiobook data pipeline for scraping, processing, and downloading Vietnamese audiobooks from multiple sources.

## Overview

The crawler module provides a complete solution for:

- Web scraping audiobook metadata from Vietnamese audiobook websites
- Batch downloading audiobooks and their corresponding text files
- Managing concurrent downloads with rate limiting
- Processing and organizing audiobook collections

## Module Structure

```
crawler/
├── download.py          # Batch downloader with async support
├── metadata.py          # Metadata extraction and processing
├── utils.py            # Web scraping utilities and helpers
├── playwright_server.py # Playwright server management
└── README.md           # This file
```

## Features

### 🚀 **High-Performance Downloading**

- Asynchronous batch downloading with configurable concurrency
- Automatic retry logic with exponential backoff
- Progress tracking with real-time statistics
- Memory-efficient streaming downloads

### 📊 **Metadata Processing**

- Extract comprehensive book metadata (title, author, narrator, duration)
- Support for multiple Vietnamese audiobook sources
- CSV export with expandable narrator information
- JSON-based intermediate storage

### 🌐 **Web Scraping**

- Playwright-based dynamic content extraction
- Multi-source URL validation
- Automatic source detection and routing
- Rate-limited concurrent requests

### 🐳 **Container Integration**

- Docker-based Playwright server management
- Automatic container lifecycle handling
- Health checking and auto-recovery

## Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install aiohttp pandas typer playwright httpx selectolax loguru rich tqdm

# Install Playwright browsers
playwright install chromium

# Ensure Docker is running for Playwright server
docker --version
```

### Basic Usage

#### 1. Extract Metadata

```bash
# Check and validate all audiobook URLs
python metadata.py --check-urls

# Fetch metadata for all books
python metadata.py --fetch-metadata

# Convert JSON metadata to CSV
python metadata.py --create-book-metadata-csv

# Fetch narrator information
python metadata.py --fetch-narrator-metadata
```

#### 2. Download Audiobooks

```bash
# Download using default configuration
python download.py

# Use custom configuration
python download.py --config-path custom_config.json --metadata-path books.csv
```

## Configuration

### Download Configuration (`download_config.json`)

```json
{
  "max_concurrent_books": 3,
  "max_concurrent_files_per_book": 8,
  "max_retries": 3,
  "chunk_size": 8192,
  "timeout": 30,
  "audio_save_path": "./data/audio/raw/",
  "text_save_path": "./data/text/pdf/"
}
```

### Pipeline Configuration (`config.json`)

```json
{
  "valid_book_urls_path": "data/valid_book_urls.txt",
  "metadata_dir": "data/metadata",
  "metadata_book_path": "data/metadata.csv",
  "metadata_narrator_path": "data/narrator_metadata.csv",
  "narrator_download_url": "https://docs.google.com/spreadsheets/d/.../export?format=csv",
  "fetch_metadata_limit": 5
}
```

## API Reference

### Downloader Class

```python
from tts_data_pipeline.crawler.download import Downloader

async with Downloader(
    max_concurrent_books=3,
    max_concurrent_files_per_book=8,
    max_retries=3
) as downloader:
    successful, failed = await downloader.download_all_books(
        metadata_df,
        audio_save_path="./audio/",
        text_save_path="./text/"
    )
```

### Metadata Extraction

```python
from tts_data_pipeline.crawler.utils import get_all_book_url

# Get all available audiobook URLs
df = await get_all_book_url()
print(f"Found {len(df)} audiobooks")
```

### Utility Functions

```python
from tts_data_pipeline.crawler.utils import (
    get_text_download_url,
    fetch_download_audio_url,
    get_web_content
)

# Get download URLs
text_url = await get_text_download_url("book_name", "thuviensach")
audio_urls = await fetch_download_audio_url("https://audiobook-url")

# Parse web content
parser = await get_web_content("https://example.com")
```

## Supported Sources

### Text Sources

- **thuviensach**: Direct PDF downloads
- **taisachhay**: Directory-based PDF access

### Audio Sources

- Multiple Vietnamese audiobook platforms
- Dynamic content extraction via Playwright
- Automatic format detection (.mp3, etc.)

## File Organization

### Directory Structure

```
data/
├── audio/
│   └── raw/
│       └── [book_name]/
│           ├── [book_name]_1.mp3
│           ├── [book_name]_2.mp3
│           └── ...
├── text/
│   └── pdf/
│       ├── book1.pdf
│       ├── book2.pdf
│       └── ...
└── metadata/
    ├── metadata.csv
    ├── narrator_metadata.csv
    └── [individual_book].json
```

### CSV Output Format

The metadata CSV includes:

- `name`, `author`, `duration`, `duration_hours`
- `text_url`, `audio_url`
- `text_download_url`, `audio_download_url`
- `narrator_1_name`, `narrator_1_url`, `narrator_2_name`, `narrator_2_url`, ...
- Quality metrics: `sample_rate`, `quality`, `word_count`, `num_sentences`
- File sizes: `audio_size`, `text_size`

## Advanced Usage

### Custom Metadata Processing

```python
import asyncio
from tts_data_pipeline.crawler.metadata import get_book_metadata

async def process_custom_books():
    semaphore = asyncio.Semaphore(5)

    book = await get_book_metadata(
        text_url=("https://text-url", "source"),
        audio_url="https://audio-url",
        semaphore=semaphore,
        save_path=Path("./metadata/")
    )

    print(f"Processed: {book.name} by {book.author}")
```

### Batch URL Validation

```python
from tts_data_pipeline.crawler.utils import check_text_url, check_audio_url

async def validate_urls(urls):
    semaphore = asyncio.Semaphore(10)
    console = Console()

    results = await asyncio.gather(*[
        check_text_url(url, console, semaphore) for url in urls
    ])

    return results
```

## Error Handling

The module includes comprehensive error handling:

- **Network timeouts**: Automatic retry with exponential backoff
- **Invalid URLs**: Graceful skipping with detailed logging
- **Container issues**: Automatic Playwright server restart
- **File system errors**: Directory creation and cleanup
- **Rate limiting**: Semaphore-based concurrency control

## Performance Optimization

### Memory Management

- Streaming downloads with configurable chunk sizes
- Bounded connection pools
- Automatic cleanup of temporary resources

### Concurrency Control

- Separate limits for books and files per book
- DNS caching and connection reuse
- Graceful handling of rate limits

### Monitoring

- Real-time progress bars with tqdm
- Detailed logging with loguru
- Rich console output for better UX

## Troubleshooting

### Common Issues

**Playwright Server Not Starting**

```bash
# Check Docker status
docker ps

# Manually start Playwright container
docker run --name playwright-server -p 3000:3000 --rm -d \
  mcr.microsoft.com/playwright:v1.51.1-noble \
  npx playwright@1.51.0 run-server --port 3000 --host 0.0.0.0
```

**Download Failures**

- Check network connectivity
- Verify metadata CSV format
- Ensure sufficient disk space
- Review timeout settings

**Metadata Extraction Issues**

- Verify website accessibility
- Check HTML structure changes
- Update user agent strings
- Adjust rate limiting

### Logging

Logs are configured with structured formatting:

```python
# Enable debug logging
logger.add("crawler.log", level="DEBUG")

# Custom log format
FORMAT_LOG = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
```

## Contributing

1. Follow the existing async/await patterns
2. Add comprehensive error handling
3. Include progress tracking for long operations
4. Update type hints and documentation
5. Test with various Vietnamese audiobook sources

## License

This module is part of the TTS Data Pipeline project. Please refer to the main project license for usage terms.
