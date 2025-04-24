from .utils import logger, align_audio_text
from tts_data_pipeline import constants
import os
import os.path as osp


def main():
  logger.info("Starting alignment...")

  # Get all audio and text files
  audio_dirs, text_dirs = (
    os.listdir(constants.AUDIO_QUALIFIED_DIR).sort(),
    os.listdir(constants.TEXT_SENTENCE_DIR).sort(),
  )

  # Check if the number of audio and text files match
  if audio_dirs != text_dirs:
    logger.error("Audio and text directories do not match.")
    return

  # Align audio and text
  for bookname in audio_dirs:
    logger.info(f"Aligning {bookname} book")

    # Setup path
    audio_path = osp.join(constants.AUDIO_QUALIFIED_DIR, bookname)
    text_path = osp.join(constants.TEXT_SENTENCE_DIR, bookname)
    os.makedirs(constants.AENEAS_OUTPUT_DIR, exist_ok=True)

    # Align each part of book
    for audio_part, text_part in zip(os.listdir(audio_path), os.listdir(text_path)):
      # Check if the number of parts between audio and text is same
      if audio_part != text_part:
        logger.error("The number of parts between audio and text do not match")

      audio_part_path = osp.join(
        audio_path, audio_part
      )  # e.g. AUDIO_QUALIFIED_DIR/bookX/bookX_1.wav
      text_part_path = osp.join(
        text_path, text_part
      )  # e.g. TEXT_SENTENCE_DIR/bookX/bookX_1.txt
      output_path = osp.join(
        constants.AENEAS_OUTPUT_DIR,
        bookname,
        audio_part.replace(".wav", ".tsv"),
      )  # e.g. AENEAS_OUTPUT_DIR/bookX/bookX_1.tsv

      # Align audio and text part
      align_audio_text(audio_part_path, text_part_path, output_path)
      logger.success(
        f"Aligned {audio_part} part of {bookname} book. \n\tOutput: {output_path}"
      )

  logger.success("Alignment completed")


if __name__ == "__main__":
  # // TODO: Build again alignment for short audio parts, then, check the result of alignment between the full text and each short audio part
  # TODO: Remove the sentence containing "CHƯƠNG XY", which XY is the chapter number

  main()
