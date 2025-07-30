# VNAVC

## TTS Data Pipeline

Check the separate README.md file which is in subfolder in `tts_data_pipeline` for more details

- Crawler Module: [README.md](tts_data_pipeline/crawler/README.md)
- Pre-Processing Module: [README.md](tts_data_pipeline/pre_processing/README.md)
- Alignment Module: [README.md](tts_data_pipeline/alignment/README.md)

## Push to Hub

A Python CLI tool for processing structured audio datasets, calculating total duration, and pushing the dataset to the Hugging Face Hub. Ideal for preparing Vietnamese TTS datasets or other speech-related corpora.

### Dataset Structure

```
dataset/
├── speaker_001/
│   ├── audio1.wav
│   ├── audio1.txt
│   ├── audio2.wav
│   └── audio2.txt
├── speaker_002/
│   ├── ...
```

Each `.wav` file must have a corresponding `.txt` transcription file with the same name.

### Installation

```bash
pip install typer librosa datasets num2words
```

### Usage

```bash
python tts_data_pipeline/push_to_hub.py
```

Options:

- `--input-dir`: Path to the input dataset directory containing speaker subdirectories.
- `--dataset-id`: Dataset ID on Hugging Face Hub.
- `--commit`: Commit message for pushing to Hugging Face Hub.
- `--private`: Make the dataset private on Hugging Face Hub.
- `--push-to-hub`: Push the dataset to Hugging Face Hub.
- `--calc-duration`: Calculate total duration and print it to the console.

## License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
