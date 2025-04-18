from .utils import logger, align_audio_text, check_dependencies
from tts_data_pipeline import constants
import os


def main():
  logger.info("Starting alignment...")

  # Check the dependencies
  if not check_dependencies():
    return

  # Get all audio and text files
  audio_files, text_files = (
    os.listdir(constants.AUDIO_QUALIFIED_DIR),
    os.listdir(constants.TEXT_SENTENCE_DIR),
  )

  # Check if the number of audio and text files match
  if len(audio_files) != len(text_files):
    logger.error("Number of audio and text files do not match.")
    return

  # Sort the audio and text files to get the pairs of files
  audio_files.sort()
  text_files.sort()

  # Align audio and text
  for audio_file, text_file in zip(audio_files, text_files):
    logger.info(f"Aligning {audio_file} with {text_file}")

    # Setup path
    audio_path = os.path.join(constants.AUDIO_QUALIFIED_DIR, audio_file)
    text_path = os.path.join(constants.TEXT_SENTENCE_DIR, text_file)
    os.makedirs(constants.ALIGNMENT_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(
      constants.ALIGNMENT_OUTPUT_DIR, audio_file.replace(".wav", ".json")
    )

    # Check if the audio and text files exist
    if not os.path.exists(audio_path):
      logger.error(f"Audio file {audio_path} does not exist.")
      continue
    if not os.path.exists(text_path):
      logger.error(f"Text file {text_path} does not exist.")
      continue

    # Run the alignment
    align_audio_text(audio_path, text_path, output_path)

  logger.success("Alignment completed")


if __name__ == "__main__":
  main()
