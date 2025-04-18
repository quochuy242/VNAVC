import glob
import os
import re
import string
from typing import List

import pdfplumber
import underthesea
from loguru import logger
from tqdm import tqdm

from tts_data_pipeline import constants

logger.remove()
logger.add(
  f"{constants.LOG_DIR}/text_processing.log",
  level="INFO",
  rotation="10 MB",
  encoding="utf-8",
  format=constants.FORMAT_LOG,
  colorize=False,
  diagnose=True,
  enqueue=True,
)


# // TODO: Build a custom Semiotic Normaliztion to normalize Vietnamese number, currency, address, date, etc.
class ViSemioticNorm:
  def __init__(self):
    # Define regex patterns
    self.number_pattern = re.compile(
      r"\b\d{1,3}(?:\.\d{3})*(?:,\d+)?|\b\d+(?:,\d+)?"
    )  # e.g., 1.000, 10,5
    self.currency_pattern = re.compile(
      r"\b\d+(?:\.\d+)?\s?(VND|vnđ|đ|USD|$|€|¥)\b", re.IGNORECASE
    )
    self.date_pattern = re.compile(
      r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b"
    )  # dd/mm/yyyy or dd-mm-yyyy
    self.address_pattern = re.compile(r"\b(Số\s\d+,\s)?(?:đường|Đường)\s[\w\s]+\b")

  def normalize_number(self, text: str) -> str:
    def replace_number(match):
      number_str = match.group().replace(".", "").replace(",", ".")
      return number_str  # You can convert to int/float or even to words

    return self.number_pattern.sub(replace_number, text)

  def normalize_currency(self, text: str) -> str:
    def replace_currency(match):
      value = match.group()
      normalized = value.replace("đ", "VND").replace("vnđ", "VND").replace("$", "USD")
      return normalized.upper()

    return self.currency_pattern.sub(replace_currency, text)

  def normalize_date(self, text: str) -> str:
    def replace_date(match):
      day, month, year = match.groups()
      normalized = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
      return normalized

    return self.date_pattern.sub(replace_date, text)

  def normalize_address(self, text: str) -> str:
    def replace_address(match):
      addr = match.group().strip()
      normalized = addr.title()
      return normalized

    return self.address_pattern.sub(replace_address, text)

  def normalize_all(self, text: str) -> str:
    text = self.normalize_currency(text)
    text = self.normalize_number(text)
    text = self.normalize_date(text)
    text = self.normalize_address(text)
    return text


def convert_pdf_to_text(pdf_path: str):
  try:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
      for page in pdf.pages:
        text += page.extract_text()
    return text
  except Exception as e:
    logger.error(f"Error processing {pdf_path}: {e}")
    return ""


def remove_punctuations(sentence: str):
  translator = str.maketrans("", "", string.punctuation)
  return sentence.translate(translator)


def process_sentence(sentence: str) -> str:
  """
  Process a sentence by removing leading and trailing spaces,
  removing punctuation, normalizing, and converting to uppercase.

  Args:
    sentence (str): Sentence to process

  Returns:
    str: Processed sentence
  """

  semiotic_norm = ViSemioticNorm()
  try:
    sentence = sentence.strip()  # remove leading and trailing spaces
    sentence = sentence.replace("D-", "Đ")  # special case for D- error
    sentence = sentence.replace(
      "https://thuviensach.vn", ""
    )  # remove link in the footer of each page
    sentence = remove_punctuations(sentence)  # remove punctuation
    sentence = underthesea.text_normalize(sentence)  # Normalize sentence (NFC)
    sentence = semiotic_norm.normalize_all(sentence)
    sentence = sentence.upper()  # convert to uppercase
  except Exception as e:
    logger.error(f"Error processing sentence: {e}")
  return sentence


def group_sentences(sentences: List[str], min_word_threshold: int = 20):
  grouped_sentences = []
  word_count = 0
  word_store = []

  for sentence in sentences:
    words: List[str] = sentence.split()
    word_count += len(words)
    word_store.extend(words)
    if word_count >= min_word_threshold:
      grouped_sentences.append(" ".join(word_store))
      word_count = 0
      word_store = []

  # WARNING: If the last sentence is not long enough, it will be dropped
  return grouped_sentences


def process_pdfs(pdf_dir: str, output_dir: str, min_word_threshold: int = 20):
  """
  Process all PDFs in a directory, extract text, split into sentences,
  normalize, and save to output file.

  Args:
    pdf_dir (str): Directory containing PDF files
    output_file (str): Path to save the normalized sentences
  """
  # Make sure the directory exists
  if not os.path.exists(pdf_dir):
    logger.warning(f"Directory {pdf_dir} does not exist")
    return

  # Get all PDF files in the directory
  pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

  # Check if any PDF files were found
  if not pdf_files:
    logger.warning(f"No PDF files found in {pdf_dir}")
    return

  # Process each PDF file with progress bar
  for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
    # Extract text from PDF
    text = convert_pdf_to_text(pdf_file)

    if text:
      # Use underthesea to split book into Vietnamese sentences
      sentences = underthesea.sent_tokenize(text)

      # TODO: Remove the sentence containing "CHƯƠNG XY", which XY is the chapter number

      # Normalize each sentence
      normalized_sentences = [process_sentence(sent) for sent in sentences]

      # Filter out empty sentences and group small consecutive sentences
      normalized_sentences = [sent for sent in normalized_sentences if sent]
      normalized_sentences = group_sentences(normalized_sentences, min_word_threshold)

    # Save all sentences to output file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
      output_dir, os.path.basename(pdf_file).replace("pdf", "txt")
    )
    with open(output_file, "w", encoding="utf-8") as f:
      for sentence in normalized_sentences:
        f.write(sentence + "\n")

  logger.success("Processing complete.")
  return


if __name__ == "__main__":
  logger.info("Starting PDF text processing...")

  process_pdfs(
    pdf_dir=constants.TEXT_PDF_DIR,
    output_dir=constants.TEXT_SENTENCE_DIR,
    min_word_threshold=constants.MIN_WORD_THRESHOLD,
  )

  logger.success("Text processing complete.")
