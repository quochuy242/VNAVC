# Pre-processing Module

This module prepares audiobook audio and text files for alignment and dataset creation. It processes raw audio (MP3) and text (PDF) files, converting, cleaning, and splitting them as needed for Text-to-Speech (TTS) pipelines.

## Features

- **Text Pre-processing:** Converts PDF books to plain text, splits into sentences, normalizes, and saves results.
- **Audio Pre-processing:** Groups audiobook parts, converts MP3 to WAV, checks sample rates, and organizes files into qualified/unqualified folders.
- **Batch Processing:** Handles all files in the specified directories automatically.

## Requirements

- Python 3.7+
- [ffmpeg](https://ffmpeg.org/) (for audio conversion)
- Python dependencies: see `requirements.txt`

## Installation

1. Clone the repository and navigate to the project directory.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure `ffmpeg` is installed and available in your PATH.

## Usage

Run the pre-processing script from the project root:

```sh
python -m tts_data_pipeline.pre_processing.main --target all
```

You can also process only text or audio:

```sh
python -m tts_data_pipeline.pre_processing.main --target text --path path/to/book.pdf
python -m tts_data_pipeline.pre_processing.main --target audio --path path/to/audio_folder/
```

### Command Line Options

- `-t`, `--target`: Type of file to process (`text`, `audio`, or `all`)
- `-p`, `--path`: Path to the file (for text) or directory (for audio) to process

## Output

- Processed text files are saved in `data/text/sentence/`
- Qualified audio files are saved in `data/audio/qualified/`
- Unqualified audio files are saved in `data/audio/unqualified/`
- Metadata is updated in `data/metadata/metadata_book.csv`

## Troubleshooting

- Check logs for error messages.
- Ensure all dependencies are installed and available in your PATH.
