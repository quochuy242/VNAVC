# Crawler module

This module is a crawler in the data pipeline for downloading audiobooks, which is from the [sachnoiviet](https://www.sachnoiviet.net/) website. It fetches a list of audiobook URLs and their corresponding text (PDF) files, which is from the [tiemsach](https://www.tiemsach.org/) website.

The crawler is designed for Text-to-Speech (TTS) applications. It offers an efficient way to build a paired text-audio dataset by crawling, downloading, and organizing content from public web sources.

## Features

- **Asynchronous Processing**: Uses `asyncio` for concurrent downloads and processing
- **Progress Tracking**: Visual progress bars for each download using `tqdm` and `rich`
- **Comprehensive Metadata**: Collects and stores metadata for each audiobook
- **Dual Content Types**: Downloads both audio files (MP3) and corresponding text files (PDF)
- **Automatic Filtering**: Ensures that only books with both audio and text available are processed

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/tts-data-pipeline.git
cd tts-data-pipeline
pip install -r requirements.txt
```

## Usage

```bash
python3 -m tts_data_pipeline.crawler
```

The pipeline performs the following steps:

1. Fetches all audiobook URLs and their names
2. Filters books to ensure they exist in the text source
3. Collects and saves metadata for each book
4. Downloads audio files and corresponding text files concurrently

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT](LICENSE)
