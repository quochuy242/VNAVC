import glob
import os

import pypdf
import underthesea
from tqdm import tqdm

import string


def remove_punctuations(sentence: str):
    translator = str.maketrans("", "", string.punctuation)
    return sentence.translate(translator)


# TODO: Build a custom Semiotic Normaliztion to normalize Vietnamese number, currency, address, date, etc.
class ViSemioticNorm:
    def __init__(self):
        pass


def convert_pdf_to_text(pdf_path: str):
    """
    Convert a PDF file to text.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text from the PDF
    """
    try:
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            text = ""

            # Extract text from each page
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()

            return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""


def process_sentence(sentence: str) -> str:
    """
    Process a sentence by removing leading and trailing spaces,
    removing punctuation, normalizing, and converting to uppercase.

    Args:
        sentence (str): Sentence to process

    Returns:
        str: Processed sentence
    """
    from tts_data_pipeline import utils

    sentence = sentence.strip()  # remove leading and trailing spaces
    sentence = utils.text.remove_punctuations(sentence)  # remove punctuation
    sentence = underthesea.normalize(sentence)  # Normalize sentence (NFC)
    sentence = sentence.upper()  # convert to uppercase
    return sentence


def process_pdfs(pdf_dir: str, output_dir: str):
    """
    Process all PDFs in a directory, extract text, split into sentences,
    normalize, and save to output file.

    Args:
        pdf_dir (str): Directory containing PDF files
        output_file (str): Path to save the normalized sentences
    """
    # Make sure the directory exists
    if not os.path.exists(pdf_dir):
        print(f"Directory {pdf_dir} does not exist")
        return

    # Get all PDF files in the directory
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))

    # Check if any PDF files were found
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    # Process each PDF file with progress bar
    for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
        # Extract text from PDF
        text = convert_pdf_to_text(pdf_file)

        if text:
            # Use underthesea to split text into Vietnamese sentences
            sentences = underthesea.sent_tokenize(text)

            # Normalize each sentence
            normalized_sentences = [process_sentence(sent) for sent in sentences]

            # Filter out empty sentences
            normalized_sentences = [sent for sent in normalized_sentences if sent]

        # Save all sentences to output file with progress bar
        output_file = os.path.join(output_dir, os.path.basename(pdf_file) + ".txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for sentence in normalized_sentences:
                f.write(sentence + "\n")

    print("Processing complete.")
    return
