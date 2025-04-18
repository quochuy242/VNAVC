# Crawler module

This module is a crawler in the data pipeline for downloading audiobooks, which is from the [sachnoiviet](https://www.sachnoiviet.net/) website. It fetches a list of audiobook URLs and their corresponding text (PDF) files, which is from the [thuviensachpdf](https://thuviensachpdf.com/) website.

The crawler is designed for Text-to-Speech (TTS) applications. It offers an efficient way to build a paired text-audio dataset by crawling, downloading, and organizing content from public web sources.

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
usage: __main__.py [-h] [-s] [-f] [--process-metadata] [-d DOWNLOAD] [--name NAME] [--author AUTHOR] [--narrator NARRATOR]

Audiobook Download Pipeline

options:
  -h, --help            show this help message and exit
  -s, --save-urls       Force to save all audiobook URLs to a file
  -f, --fetch-metadata  Force to fetch metadata for each book
  --process-metadata    Process and convert metadata files to a single CSV file
  -d DOWNLOAD, --download DOWNLOAD
                        Download books (available: all, none, query) (default: none)
  --name NAME           Download books by name when --download is query
  --author AUTHOR       Download books by author when --download is query
  --narrator NARRATOR   Download books by narrator when --download is query
```

You can combine multiple operations in a single command:

```bash
python3 -m tts_data_pipeline.crawler --save-urls --fetch-metadata --process-metadata --download "all"
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
