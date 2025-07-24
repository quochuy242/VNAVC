# Audio-Text Alignment Tool

A Python-based tool for aligning audio files with their corresponding text transcripts using the Aeneas library. This tool supports batch processing, parallel execution, and automatic segmentation of aligned audio-text pairs.

## Features

- **Audio-Text Alignment**: Uses the Aeneas library for precise forced alignment
- **Batch Processing**: Process multiple audio-text pairs simultaneously
- **Parallel Execution**: Multi-threaded processing for improved performance
- **Automatic Segmentation**: Split aligned files into individual segments
- **Outlier Detection**: Remove segments that are too short or too long
- **Rich CLI Interface**: Interactive command-line interface with progress bars
- **Configurable Parameters**: Flexible configuration via JSON files or CLI options

## Prerequisites

### System Dependencies

The following system dependencies must be installed:

- **FFmpeg**: For audio processing
- **FFprobe**: For audio file analysis (usually included with FFmpeg)
- **eSpeak**: For text-to-speech synthesis used by Aeneas

#### Installation on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg espeak espeak-data
```

#### Installation on macOS:

```bash
brew install ffmpeg espeak
```

#### Installation on Windows:

- Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- Download eSpeak from [http://espeak.sourceforge.net/download.html](http://espeak.sourceforge.net/download.html)
- Add both to your system PATH

### Python Dependencies

```bash
pip install aeneas pandas typer loguru rich pydantic typing-extensions
```

## Installation

1. Clone or download the script
2. Install system dependencies (see above)
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The tool provides three main commands: `create-config`, `test`, and `run`.

### 1. Create Configuration File

Generate a sample configuration file:

```bash
python audio_text_aligner.py create-config --output-path config.json
```

This creates a `config.json` file with default settings that you can customize:

```json
{
  "audio_dir": "data/audio/processed",
  "text_dir": "data/text/processed",
  "align_dir": "data/alignment",
  "log_dir": "logs",
  "remove_outliers": true,
  "split": false,
  "max_workers": 8,
  "min_duration": 3.0,
  "max_duration": 12.0
}
```

### 2. Test Single File Alignment

Test alignment on a single audio-text pair:

```bash
python audio_text_aligner.py test \
  --audio path/to/audio.wav \
  --text path/to/text.txt \
  --split \
  --max-workers 4
```

#### Parameters:

- `--audio, -a`: Path to the audio file (required)
- `--text, -t`: Path to the text file (required)
- `--split, -s`: Enable splitting into segments (optional)
- `--align-output, -o`: Custom output path for alignment file (optional)
- `--max-workers`: Number of parallel workers (default: CPU count)

### 3. Batch Processing

Process multiple audio-text pairs:

```bash
python audio_text_aligner.py run \
  --config config.json \
  --audio-dir data/audio \
  --text-dir data/text \
  --split \
  --max-workers 8
```

#### Parameters:

- `--config, -c`: Path to configuration file
- `--audio-dir, -d`: Directory containing audio files
- `--text-dir, -t`: Directory containing text files
- `--split, -s`: Enable splitting into segments
- `--max-workers`: Number of parallel workers
- `--remove-outliers, -r`: Remove segments outside duration range

## File Requirements

### Audio Files

- **Format**: WAV files only (`.wav`)
- **Quality**: 16-bit PCM recommended
- **Sample Rate**: Any standard rate (22050 Hz, 44100 Hz, etc.)

### Text Files

- **Format**: Plain text files (`.txt`)
- **Encoding**: UTF-8
- **Content**: One sentence/phrase per line
- **Naming**: Must match corresponding audio file (e.g., `book1.wav` ↔ `book1.txt`)

## Configuration Options

| Parameter         | Type  | Default                | Description                            |
| ----------------- | ----- | ---------------------- | -------------------------------------- |
| `audio_dir`       | Path  | `data/audio/processed` | Directory containing audio files       |
| `text_dir`        | Path  | `data/text/processed`  | Directory containing text files        |
| `align_dir`       | Path  | `data/alignment`       | Output directory for alignment results |
| `log_dir`         | Path  | `logs`                 | Directory for log files                |
| `remove_outliers` | bool  | `true`                 | Remove segments outside duration range |
| `split`           | bool  | `false`                | Split files into individual segments   |
| `max_workers`     | int   | CPU count              | Maximum parallel workers               |
| `min_duration`    | float | `3.0`                  | Minimum segment duration (seconds)     |
| `max_duration`    | float | `12.0`                 | Maximum segment duration (seconds)     |

## Output Structure

### Alignment Files

The tool generates alignment files in TSV format with columns:

- `start`: Start time in seconds
- `end`: End time in seconds
- `id`: Line/segment ID
- `duration`: Segment duration in seconds

### Segmented Files (when `split=true`)

```
dataset/
├── narrator_id/
│   ├── book1_0.wav
│   ├── book1_0.txt
│   ├── book1_1.wav
│   ├── book1_1.txt
│   └── ...
└── outlier.txt
```

### Log Files

- `logs/alignment.log`: Detailed processing logs
- Console output with rich formatting and progress bars

## Examples

### Basic Single File Test

```bash
python audio_text_aligner.py test \
  -a samples/chapter1.wav \
  -t samples/chapter1.txt
```

### Test with Segmentation

```bash
python audio_text_aligner.py test \
  -a samples/chapter1.wav \
  -t samples/chapter1.txt \
  --split \
  --max-workers 4
```

### Batch Processing with Custom Config

```bash
python audio_text_aligner.py run \
  --config my_config.json \
  --split \
  --remove-outliers
```

### Override Config Settings

```bash
python audio_text_aligner.py run \
  --config config.json \
  --audio-dir /path/to/audio \
  --text-dir /path/to/text \
  --max-workers 16 \
  --split
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies Error**

   ```
   Missing dependencies: ffmpeg, espeak. Please install them.
   ```

   **Solution**: Install system dependencies as described in Prerequisites

2. **File Not Found**

   ```
   Audio file not found: path/to/file.wav
   ```

   **Solution**: Verify file paths and ensure files exist

3. **Unsupported Format**

   ```
   Unsupported audio format: .mp3
   ```

   **Solution**: Convert audio files to WAV format

4. **Memory Issues with Large Files**
   **Solution**: Reduce `max_workers` or process files individually

### Performance Tips

- Use WAV files with appropriate sample rates (22050 Hz for speech)
- Limit `max_workers` based on available RAM
- Process files in smaller batches for very large datasets
- Use SSD storage for better I/O performance

## Dependencies

### Python Packages

- `aeneas`: Forced alignment library
- `pandas`: Data manipulation
- `typer`: CLI framework
- `loguru`: Logging
- `rich`: Terminal formatting
- `pydantic`: Configuration validation

### System Requirements

- Python 3.7+
- FFmpeg
- eSpeak
- Sufficient RAM for parallel processing

## License

This tool is designed for research and educational purposes. Please ensure compliance with the licenses of all dependencies, particularly the Aeneas library.

## Contributing

When contributing:

1. Ensure all system dependencies are documented
2. Add appropriate error handling for new features
3. Update configuration options and CLI help text
4. Include example usage for new functionality

## Support

For issues related to:

- **Aeneas**: Check the [Aeneas documentation](https://github.com/readbeyond/aeneas)
- **Audio processing**: Verify FFmpeg installation and file formats
- **Performance**: Adjust `max_workers` and batch sizes based on system resources
