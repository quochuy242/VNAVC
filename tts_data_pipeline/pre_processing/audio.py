import contextlib
import os
import shutil
import wave

from pydub import AudioSegment
from tqdm import tqdm

from tts_data_pipeline import constants


def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
    """
    Convert an MP3 file to WAV format

    Args:
        mp3_path (str): Path to the MP3 file
        wav_path (str): Path where the WAV file will be saved
    """
    try:
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format="wav")
        return True
    except Exception as e:
        print(f"Error converting {mp3_path}: {e}")
        return False


def get_wav_sample_rate(wav_path: str):
    """
    Get the sample rate of a WAV file

    Args:
        wav_path (str): Path to the WAV file

    Returns:
        int: Sample rate in Hz, or None if there was an error
    """
    try:
        with contextlib.closing(wave.open(wav_path, "r")) as wf:
            sample_rate = wf.getframerate()
            return sample_rate
    except Exception as e:
        print(f"Error getting sample rate for {wav_path}: {e}")
        return None


def process_audio_files(
    mp3_dir: str,
    qualified_dir: str,
    unqualified_dir: str,
    min_sample_rate: int = constants.MIN_SAMPLE_RATE,
):
    """
    Process all MP3 files in a directory:
    1. Convert them to WAV format
    2. Check their sample rate
    3. Move those with sample rates below the threshold to unqualified folder

    Args:
        mp3_dir (str): Directory containing MP3 files
        qualified_dir (str): Directory to save converted WAV files
        unqualified_dir (str): Directory to move unqualified files
        min_sample_rate (int): Minimum acceptable sample rate in Hz
    """
    # Create output directories if they don't exist
    os.makedirs(qualified_dir, exist_ok=True)
    os.makedirs(unqualified_dir, exist_ok=True)

    # Get all MP3 files in the directory
    mp3_files = [f for f in os.listdir(mp3_dir) if f.lower().endswith(".mp3")]

    if not mp3_files:
        print(f"No MP3 files found in {mp3_dir}")
        return

    qualified_count = 0
    unqualified_count = 0

    # Process each MP3 file with progress bar
    for mp3_file in tqdm(mp3_files, desc="Processing audio files"):
        mp3_path = os.path.join(mp3_dir, mp3_file)
        wav_filename = (
            os.path.splitext(mp3_file)[0] + ".wav"
        )  # Change extension to .wav
        wav_path = os.path.join(qualified_dir, wav_filename)

        # Convert MP3 to WAV
        if convert_mp3_to_wav(mp3_path, wav_path):
            # Check sample rate
            sample_rate = get_wav_sample_rate(wav_path)

            if sample_rate is not None:
                if sample_rate < min_sample_rate:
                    # Move to unqualified directory
                    unqualified_path = os.path.join(unqualified_dir, wav_filename)
                    shutil.move(wav_path, unqualified_path)
                    print(
                        f"Moved {wav_filename} to unqualified folder (sample rate: {sample_rate} Hz)"
                    )
                    unqualified_count += 1
                else:
                    # Move to qualified directory
                    print(f"Qualified: {wav_filename} (sample rate: {sample_rate} Hz)")
                    qualified_count += 1

    print("\nProcessing complete:")
    print(f"- Total files processed: {len(mp3_files)}")
    print(f"- Qualified files (≥ {min_sample_rate} Hz): {qualified_count}")
    print(f"- Unqualified files (< {min_sample_rate} Hz): {unqualified_count}")
