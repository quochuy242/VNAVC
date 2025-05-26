import os
import os.path as osp
import re
import string
from typing import List, Optional

import pandas as pd
import pdfplumber
import underthesea
from loguru import logger

from tts_data_pipeline import constants

logger.remove()
logger.add(
  f"{constants.LOG_DIR}/pre_processing.log",
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


def convert_pdf_to_text(pdf_path: str) -> Optional[str]:
  """
  Convert a PDF file to text using pdfplumber

  Args:
      pdf_path (str): Path to the PDF file

  Returns:
      str | None: Text extracted from the PDF
  """
  try:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
      for page in pdf.pages:
        text += " "
        text += page.extract_text()
    return text
  except Exception as e:
    logger.exception(f"Error processing {pdf_path}: {e}")
    return None


def count_word(text: str) -> int:
  """
  Count the number of words in a enormous string
  """
  in_word = False
  count = 0
  for char in text:
    if char.isspace():
      in_word = False
    elif not in_word:
      count += 1
      in_word = True
  return count


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
    sentence = sentence.lower()  # convert to lowercase
  except Exception as e:
    logger.exception(f"Error processing sentence: {e}")
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


# CAUTION: Not used
def split_textbook(sentences: List[str], book_name: str):
  # Get the number of part of audiobook
  audiobook_path = osp.join(constants.AUDIO_QUALIFIED_DIR, book_name)
  if osp.exists(audiobook_path):
    num_of_part = len(os.listdir(audiobook_path))
  else:
    num_of_part = 1
    logger.error(f"The {book_name} don't exist")

  num_sentence_each_part = len(sentences) // num_of_part
  return [
    sentences[i * num_sentence_each_part : (i + 1) * num_sentence_each_part]
    for i in range(num_of_part + 1)
  ], num_of_part


def text_processing(
  pdf_path: str,
  min_word_threshold: int,
  update_metadata: bool = False,
  save_book: bool = True,
  remove_original_files: bool = True,
):
  """
  Process a single PDF file containing extracting text, splitting into sentences,
  normalizing, and saving to output file.

  Args:
    pdf_path (str): Directory containing PDF files
    min_word_threshold (int): Minimum number of words in a sentence
    update_metadata (bool): Whether to update metadata file
    save_book (bool): Whether to save book file
  """

  # Read metadata file
  metadata_df = (
    pd.read_csv(constants.METADATA_BOOK_PATH) if update_metadata else pd.DataFrame()
  )

  # Extract text from PDF
  text = convert_pdf_to_text(pdf_path)
  pdf_filename = osp.splitext(osp.basename(pdf_path))[0]

  if text:
    # Use underthesea to split book into Vietnamese sentences
    sentences = underthesea.sent_tokenize(text)

    # Normalize each sentence
    normalized_sentences = [process_sentence(sent) for sent in sentences]

    # Filter out empty sentences and group small consecutive sentences
    normalized_sentences = [sent for sent in normalized_sentences if sent]
    normalized_sentences = group_sentences(normalized_sentences, min_word_threshold)

    # Save all sentences to output file
    output_path = osp.join(constants.TEXT_SENTENCE_DIR, pdf_filename + ".txt")
    if save_book:
      os.makedirs(osp.dirname(output_path), exist_ok=True)
      with open(output_path, "w", encoding="utf-8") as f:
        for sentence in normalized_sentences:
          f.write(sentence + "\n")

    if update_metadata:
      # Update word count
      metadata_df.loc[
        metadata_df["text_url"].str.contains(pdf_filename), "word_count"
      ] = count_word(text)

      # Update number of sentences
      metadata_df.loc[
        metadata_df["text_url"].str.contains(pdf_filename), "num_sentences"
      ] = len(normalized_sentences)

      # Save metadata
      metadata_df.to_csv(constants.METADATA_BOOK_PATH, index=False)

    # Remove original file
    if remove_original_files:
      os.remove(pdf_path)

  return pdf_filename
