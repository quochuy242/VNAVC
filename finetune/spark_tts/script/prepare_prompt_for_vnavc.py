import argparse
import csv
import json
import os
import shutil

from datasets import load_dataset, Dataset
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from tqdm import tqdm

from huggingface_hub import HfApi


class AudioPromptDataset:
  def __init__(self, model_name_or_path, device):
    self.audio_tokenizer = BiCodecTokenizer(model_name_or_path, device=device)

  def tokenize(self, data_dir, output_dir):
    """
    Tokenizes the audio dataset based on LJSpeech format and saves it as a JSONL file.

    Args:
      data_dir (str): Path to the dataset directory in LJSpeech format.
      output_dir (str): Path to save the tokenized JSONL output.
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_filename = os.path.basename(data_dir) + ".jsonl"
    jsonl_path = os.path.join(output_dir, jsonl_filename)

    print(f"Tokenizing audio and saving to {jsonl_path}...")
    with open(jsonl_path, "w") as f:
      with open(
        os.path.join(data_dir, "metadata.csv"), mode="r", encoding="utf-8"
      ) as file:
        reader = csv.reader(file, delimiter="|")
        for row in tqdm(reader):
          try:
            audio_name, text, _ = row
          except Exception as e:
            print(f"Error parsing row: {row}. {e}")
            audio_name, text = row

          # Skip header or malformed rows
          if not audio_name or not text:
            continue

          audio_path = os.path.join(data_dir, "wavs", audio_name + ".wav")
          if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found at {audio_path}. Skipping.")
            continue

          global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
            audio_path
          )
          global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
          )
          semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
          )
          inputs = [
            "<|task_tts|>",
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
            "<|start_semantic_token|>",
            semantic_tokens,
            "<|end_semantic_token|>",
            "<|im_end|>",
          ]
          inputs = "".join(inputs)
          prompt = {"text": inputs}
          f.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    print(f"Tokenization complete. Output saved to {jsonl_path}")
    return jsonl_path


def download_and_process_vnavc(output_dir):
  """
  Downloads the qhuy242/VNAVC dataset and formats it like LJSpeech.

  Args:
    output_dir (str): The base directory to save the formatted dataset.

  Returns:
    str: The path to the newly created dataset directory.
  """
  print("Downloading and processing the qhuy242/VNAVC dataset...")
  # Load the dataset from Hugging Face Hub
  dataset = load_dataset("qhuy242/VNAVC", split="train")

  # Create the LJSpeech-style directory structure
  dataset_dir = os.path.join(output_dir, "VNAVC_LJSpeech_format")
  wavs_dir = os.path.join(dataset_dir, "wavs")
  os.makedirs(wavs_dir, exist_ok=True)

  # Create the metadata.csv file
  metadata_path = os.path.join(dataset_dir, "metadata.csv")
  with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile, delimiter="|")
    for item in tqdm(dataset):
      audio_path = item["audio"]["path"]
      text = item["sentence"]

      # Use the filename as the unique identifier
      audio_filename = os.path.basename(audio_path).replace(".wav", "")

      # Copy the audio file to the new wavs directory
      shutil.copy(audio_path, os.path.join(wavs_dir, audio_filename + ".wav"))

      # Write the metadata row: filename|text|
      writer.writerow([audio_filename, text, ""])

  print(f"Dataset successfully formatted in LJSpeech style at {dataset_dir}")
  return dataset_dir


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Tokenize audio dataset for TTS and push to Hugging Face Hub."
  )
  parser.add_argument(
    "--data_dir",
    type=str,
    default="VNAVC_raw_data",
    help="Path to the directory where the VNAVC dataset will be downloaded and processed.",
  )
  parser.add_argument(
    "--output_dir",
    type=str,
    default="output_prompt",
    help="Path to save the tokenized output.",
  )
  parser.add_argument(
    "--push_prompt_to_hub",
    action="store_true",
    help="Push the tokenized output to Hugging Face Hub.",
  )
  parser.add_argument(
    "--dataset_id",
    type=str,
    help="The Hugging Face dataset ID to push to (e.g., 'your-username/your-dataset-name'). Required if --push_prompt_to_hub is set.",
  )
  parser.add_argument(
    "--commit_message",
    type=str,
    default="Add VNAVC tokenized data",
    help="The commit message to use when pushing to the Hub.",
  )
  parser.add_argument(
    "--private",
    action="store_true",
    help="Whether to create a private repository on the Hub.",
  )

  args = parser.parse_args()

  # First, download and process the VNAVC dataset to the specified data_dir
  ljspeech_formatted_dir = download_and_process_vnavc(args.data_dir)

  # Then, tokenize the formatted dataset
  processor = AudioPromptDataset(
    model_name_or_path="pretrained_models/Spark-TTS-0.5B", device="cuda"
  )
  jsonl_path = processor.tokenize(ljspeech_formatted_dir, args.output_dir)

  # If the flag is set, push the tokenized output to the Hugging Face Hub
  if args.push_prompt_to_hub:
    if not args.dataset_id:
      raise ValueError(
        "Argument --dataset_id is required when --push_prompt_to_hub is set."
      )

    print(f"\nPushing tokenized dataset to Hugging Face Hub as '{args.dataset_id}'...")
    print("Remember to log in with `huggingface-cli login` if you haven't already.")

    # Load the tokenized JSONL file into a datasets.Dataset object
    # The file path is a list because load_dataset accepts a list of files.
    ds = load_dataset("json", data_files={"train": jsonl_path}, split="train")

    # Push the dataset to the specified Hugging Face repository
    # The 'private' argument is passed to make the repo private if the flag is set
    ds.push_to_hub(
      repo_id=args.dataset_id,
      commit_message=args.commit_message,
      private=args.private,
    )

    print(
      f"Dataset successfully pushed to https://huggingface.co/datasets/{args.dataset_id}"
    )
