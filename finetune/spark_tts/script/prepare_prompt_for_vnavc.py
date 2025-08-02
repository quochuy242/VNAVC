import argparse
import csv
import json
import os
import shutil
import torch
import gc

import soundfile as sf
from datasets import load_dataset
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from tqdm import tqdm


class AudioPromptDataset:
  def __init__(self, model_name_or_path, device):
    """
    Initializes AudioPromptDataset with the BiCodecTokenizer model.
    Args:
      model_name_or_path (str): The path to the audio tokenizer model.
      device (str): The device to run the model on (e.g., 'cuda', 'cpu').
    """
    self.audio_tokenizer = BiCodecTokenizer(model_name_or_path, device=device)

  def tokenize(self, data_dir, output_dir, jsonl_file_path, subset_name="subset"):
    """
    Tokenizes audio files in a specific directory and appends them to a JSONL file.

    Args:
      data_dir (str): Path to the directory containing the subset's .wav files and metadata.csv.
      output_dir (str): Path to save the JSONL file.
      jsonl_file_path (str): The full path of the output JSONL file.
      subset_name (str): The name of the current subset for printing.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Tokenizing audio and appending to {jsonl_file_path} for {subset_name}...")

    # Open the JSONL file in 'a' (append) mode to add data to the end
    with open(jsonl_file_path, "a") as f:
      metadata_path = os.path.join(data_dir, "metadata.csv")
      if not os.path.exists(metadata_path):
        print(
          f"Warning: metadata.csv not found in {data_dir}. Skipping tokenization for this subset."
        )
        return

      with open(metadata_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="|")
        for row in tqdm(reader):
          try:
            # If the speaker_id is missing, skip this row
            audio_name, text, *_ = row
          except Exception as e:
            print(f"Error parsing row: {row}. {e}")
            continue

          if not audio_name or not text:
            continue

          audio_path = os.path.join(data_dir, "wavs", audio_name)
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

    print(f"Tokenization complete for {subset_name}.")

    # Delete the temporary directory to free up disk space
    print(f"Deleting temporary directory: {data_dir}...")
    shutil.rmtree(data_dir)
    print("Deletion successful.")


def process_subset(dataset_subset, output_dir, subset_name):
  """
  Downloads and processes a subset of the VNAVC dataset, formatting it in LJSpeech style.

  Args:
    dataset_subset (Dataset): A subset of the dataset.
    output_dir (str): The directory to save the temporary files for the subset.
    subset_name (str): The name of the subset for printing.

  Returns:
    str: The path to the temporary directory of the subset.
  """
  print(f"\nProcessing {len(dataset_subset)} samples for {subset_name}...")

  # Create the LJSpeech-style directory structure
  dataset_dir = os.path.join(output_dir, subset_name)
  wavs_dir = os.path.join(dataset_dir, "wavs")
  os.makedirs(wavs_dir, exist_ok=True)

  # Create the metadata.csv file
  metadata_path = os.path.join(dataset_dir, "metadata.csv")
  with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile, delimiter="|")

    for item in tqdm(dataset_subset):
      audio_arr = item["audio"]["array"]
      audio_sr = item["audio"]["sampling_rate"]
      text = item["text"]
      speaker_id = item["speaker_id"]
      new_filename = os.path.basename(item["audio"]["path"])

      new_audio_path = os.path.join(wavs_dir, new_filename)

      try:
        sf.write(new_audio_path, audio_arr, audio_sr, format="WAV")
      except sf.LibsndfileError as e:
        print(f"Error saving {new_audio_path}: {e}")
        continue

      # Write a metadata row to the CSV file
      writer.writerow([new_filename, text, speaker_id])

  return dataset_dir


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Tokenize audio dataset for TTS and push to Hugging Face Hub."
  )
  parser.add_argument(
    "--data_dir",
    type=str,
    default="VNAVC_raw_data",
    help="Path to the directory where temporary subset data will be stored.",
  )
  parser.add_argument(
    "--output_dir",
    type=str,
    default="output_prompt",
    help="Path to save the final tokenized JSONL output.",
  )
  parser.add_argument(
    "--push_prompt_to_hub",
    action="store_true",
    help="Push the tokenized output to Hugging Face Hub.",
  )
  parser.add_argument(
    "--dataset_id",
    type=str,
    help="The Hugging Face dataset ID to push to.",
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
  parser.add_argument(
    "--num_samples_per_subset",
    type=int,
    default=500,  # Number of samples to process in each subset
    help="Number of samples to process in each subset.",
  )

  args = parser.parse_args()

  # Create the output directory if it doesn't exist
  os.makedirs(args.output_dir, exist_ok=True)
  jsonl_path = os.path.join(args.output_dir, "tokenized_vnavc.jsonl")

  # Ensure the output JSONL file is empty before starting
  if os.path.exists(jsonl_path):
    os.remove(jsonl_path)

  print("Loading the full VNAVC dataset from Hugging Face Hub...")
  dataset = load_dataset("qhuy242/VNAVC", split="train")
  total_samples = len(dataset)
  num_subsets = (
    total_samples + args.num_samples_per_subset - 1
  ) // args.num_samples_per_subset

  print(f"Total samples: {total_samples}")
  print(f"Number of samples per subset: {args.num_samples_per_subset}")
  print(f"Total subsets to process: {num_subsets}")

  # Initialize the tokenizer only once
  processor = AudioPromptDataset(
    model_name_or_path="pretrained_models/Spark-TTS-0.5B", device="cuda"
  )

  for i in range(num_subsets):
    start_index = i * args.num_samples_per_subset
    end_index = min((i + 1) * args.num_samples_per_subset, total_samples)
    subset_name = f"subset_{i + 1:02d}"

    # Select the current subset
    dataset_subset = dataset.select(range(start_index, end_index))

    # Process and tokenize the subset
    ljspeech_formatted_dir = process_subset(dataset_subset, args.data_dir, subset_name)
    processor.tokenize(ljspeech_formatted_dir, args.output_dir, jsonl_path, subset_name)

    # Clean up GPU memory after each subset
    del ljspeech_formatted_dir, dataset_subset
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    gc.collect()

  print("\nAll subsets have been processed. Pipeline complete.")

  # If the flag is set, push the tokenized data to the Hugging Face Hub
  if args.push_prompt_to_hub:
    if not args.dataset_id:
      raise ValueError(
        "Argument --dataset_id is required when --push_prompt_to_hub is set."
      )
    print(f"\nPushing tokenized dataset to Hugging Face Hub as '{args.dataset_id}'...")
    print("Remember to log in with `huggingface-cli login` if you haven't already.")

    ds = load_dataset("json", data_files={"train": jsonl_path}, split="train")
    ds.push_to_hub(
      repo_id=args.dataset_id,
      commit_message=args.commit_message,
      private=args.private,
    )

    print(
      f"Dataset successfully pushed to https://huggingface.co/datasets/{args.dataset_id}"
    )
