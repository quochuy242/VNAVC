# Crawler module

This module is a crawler in the data pipeline for downloading audiobooks, which is from the [sachnoiviet](https://www.sachnoiviet.net/) website. It fetches a list of audiobook URLs and their corresponding text (PDF) files, which is from the [thuviensachpdf](https://thuviensachpdf.com/) website.

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
# Clone the repository
git clone https://github.com/quochuy242/tts-data-pipeline.git
cd tts-data-pipeline

# Activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

This script provides an asynchronous pipeline to download audiobooks, fetch metadata, and process the data efficiently.

### Command Line Options

Run the script with various options to control the workflow:

```
$ python3 -m tts_data_pipeline.crawler --help

usage: __main__.py [-h] [--save-urls] [--fetch-metadata] [--convert-metadata] [--download-books]

Audiobook Download Pipeline

options:
  -h, --help          show this help message and exit
  --save-urls         Saving all audiobook URLs to a file
  --fetch-metadata    Fetch metadata for each book
  --convert-metadata  Convert metadata to CSV
  --download-books    Download books
```

You can combine multiple operations in a single command:

```bash
python3 -m tts_data_pipeline.crawler --save-urls --fetch-metadata --convert-metadata --download-books
```

### Expected Output

- Audiobook URLs are stored in the specified path.
- Metadata is saved in JSON format.
- Converted metadata is available as a CSV file.
- Audiobooks are downloaded and stored in the appropriate directories.

For troubleshooting or additional configurations, check the `constants.py` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT](LICENSE)
