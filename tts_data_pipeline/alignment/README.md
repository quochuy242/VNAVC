# Alignment Module

This module aligns audiobook audio files with their corresponding text files using [Aeneas](https://www.readbeyond.it/aeneas/). It splits the audio and text into aligned segments for use in Text-to-Speech (TTS) datasets.

## Requirements

- Python 3.7+
- [ffmpeg](https://ffmpeg.org/)
- [ffprobe](https://ffmpeg.org/ffprobe.html)
- [espeak](http://espeak.sourceforge.net/)
- Python dependencies: see `requirements.txt`

## Installation

1. Clone the repository and navigate to the project directory.
2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure `ffmpeg`, `ffprobe`, and `espeak` are installed and available in your PATH.

## Usage

Run the alignment script from the project root:

```sh
python -m tts_data_pipeline.alignment.main \
  --audio path/to/audio.wav \
  --text path/to/text.txt \
  --split \
  --jobs 4
```

### Command Line Options

- `-a`, `--audio`: Path to the audio file to align (required)
- `-t`, `--text`: Path to the text file to align (required)
- `-s`, `--split`: Split audio and text into aligned segments (optional)
- `-j`, `--jobs`: Number of parallel jobs to use (`-1` for all cores, default: 1)
- `-f`, `--force`: Force re-alignment even if alignment data exists

## Output

- Alignment results are saved in the directory specified by `constants.AENEAS_OUTPUT_DIR`.
- Split audio and text segments are saved in the dataset directory.

## Troubleshooting

- Check `logs/alignment.log` for detailed logs.
- Ensure all dependencies are installed and available in your PATH.
