import argparse
import csv
import gc
import json
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import soundfile as sf
import torch
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
    self.write_lock = Lock()  # Thread-safe file writing

  def _process_audio_file(self, audio_path, text):
    """Process a single audio file and return the formatted prompt."""
    try:
      global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(audio_path)

      # Pre-allocate lists for better performance
      global_tokens = "".join(
        f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()
      )
      semantic_tokens = "".join(
        f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()
      )

      # Use list join instead of string concatenation
      inputs_list = [
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
      inputs = "".join(inputs_list)

      # Clean up immediately
      del global_token_ids, semantic_token_ids, global_tokens, semantic_tokens

      return {"text": inputs}
    except Exception as e:
      print(f"Error processing {audio_path}: {e}")
      return None

  @torch.inference_mode()
  def tokenize(
    self,
    data_dir,
    output_dir,
    jsonl_file_path,
    subset_name="subset",
    batch_size=8,
    num_workers=4,
  ):
    """
    Tokenizes audio files in batches with parallel I/O operations.

    Args:
        data_dir (str): Path to the directory containing the subset's .wav files and metadata.csv.
        output_dir (str): Path to save the JSONL file.
        jsonl_file_path (str): The full path of the output JSONL file.
        subset_name (str): The name of the current subset for printing.
        batch_size (int): Number of files to process in each batch.
        num_workers (int): Number of worker threads for I/O operations.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Tokenizing audio and appending to {jsonl_file_path} for {subset_name}...")

    metadata_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
      print(
        f"Warning: metadata.csv not found in {data_dir}. Skipping tokenization for this subset."
      )
      return

    # Read all metadata first
    audio_files = []
    with open(metadata_path, mode="r", encoding="utf-8") as file:
      reader = csv.reader(file, delimiter="|")
      for row in reader:
        try:
          audio_name, text, *_ = row
          if not audio_name or not text:
            continue

          audio_path = os.path.join(data_dir, "wavs", audio_name)
          if os.path.exists(audio_path):
            audio_files.append((audio_path, text))
          else:
            print(f"Warning: Audio file not found at {audio_path}. Skipping.")
        except Exception as e:
          print(f"Error parsing row: {row}. {e}")
          continue

    # Process in batches
    total_files = len(audio_files)
    processed_count = 0

    with open(
      jsonl_file_path, "a", buffering=8192
    ) as f:  # Larger buffer for better I/O
      with tqdm(total=total_files, desc=f"Processing {subset_name}") as pbar:
        for i in range(0, total_files, batch_size):
          batch = audio_files[i : i + batch_size]
          batch_results = []

          # Process batch on GPU
          for audio_path, text in batch:
            result = self._process_audio_file(audio_path, text)
            if result:
              batch_results.append(result)

          # Write batch results to file
          for result in batch_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

          processed_count += len(batch)
          pbar.update(len(batch))

          # Memory cleanup after each batch
          if i % (batch_size * 4) == 0:  # Every 4 batches
            torch.cuda.empty_cache()
            gc.collect()

    print(
      f"Tokenization complete for {subset_name}. Processed {processed_count} files."
    )

    # Delete the temporary directory to free up disk space
    print(f"Deleting temporary directory: {data_dir}...")
    shutil.rmtree(data_dir)
    print("Deletion successful.")


def write_audio_file(args):
  """Helper function for parallel audio file writing."""
  item, wavs_dir = args
  try:
    audio_arr = item["audio"]["array"]
    audio_sr = item["audio"]["sampling_rate"]
    new_filename = os.path.basename(item["audio"]["path"])
    new_audio_path = os.path.join(wavs_dir, new_filename)

    sf.write(new_audio_path, audio_arr, audio_sr, format="WAV")
    return (new_filename, item["text"], item["speaker_id"])
  except Exception as e:
    print(f"Error processing audio file: {e}")
    return None


def process_subset(dataset_subset, output_dir, subset_name, num_workers=None):
  """
  Downloads and processes a subset of the VNAVC dataset with parallel I/O operations.

  Args:
      dataset_subset (Dataset): A subset of the dataset.
      output_dir (str): The directory to save the temporary files for the subset.
      subset_name (str): The name of the subset for printing.
      num_workers (int): Number of worker processes for parallel I/O.

  Returns:
      str: The path to the temporary directory of the subset.
  """
  if num_workers is None:
    num_workers = min(mp.cpu_count(), 8)  # Limit to avoid overwhelming the system

  print(
    f"\nProcessing {len(dataset_subset)} samples for {subset_name} with {num_workers} workers..."
  )

  # Create the LJSpeech-style directory structure
  dataset_dir = os.path.join(output_dir, subset_name)
  wavs_dir = os.path.join(dataset_dir, "wavs")
  os.makedirs(wavs_dir, exist_ok=True)

  # Prepare arguments for parallel processing
  audio_write_args = [(item, wavs_dir) for item in dataset_subset]

  # Process audio files in parallel
  metadata_rows = []
  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    # Submit all tasks
    future_to_args = {
      executor.submit(write_audio_file, args): args for args in audio_write_args
    }

    # Collect results with progress bar
    with tqdm(
      total=len(audio_write_args), desc=f"Writing audio files for {subset_name}"
    ) as pbar:
      for future in as_completed(future_to_args):
        result = future.result()
        if result:
          metadata_rows.append(result)
        pbar.update(1)

  # Write metadata.csv
  metadata_path = os.path.join(dataset_dir, "metadata.csv")
  with open(
    metadata_path, "w", newline="", encoding="utf-8", buffering=8192
  ) as csvfile:
    writer = csv.writer(csvfile, delimiter="|")
    writer.writerows(metadata_rows)

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
    default=500,
    help="Number of samples to process in each subset.",
  )
  parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    help="Batch size for GPU processing.",
  )
  parser.add_argument(
    "--num_workers",
    type=int,
    default=None,
    help="Number of worker threads/processes for I/O operations.",
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
  print(f"Batch size: {args.batch_size}")
  print(f"Number of workers: {args.num_workers or 'auto'}")

  # Pre-warm GPU (optional but can help with consistent timing)
  print("Pre-warming GPU...")
  dummy_processor = AudioPromptDataset(
    model_name_or_path="pretrained_models/Spark-TTS-0.5B", device="cuda"
  )
  del dummy_processor
  torch.cuda.empty_cache()
  gc.collect()

  for i in range(num_subsets):
    start_index = i * args.num_samples_per_subset
    end_index = min((i + 1) * args.num_samples_per_subset, total_samples)
    subset_name = f"subset_{i + 1:02d}"

    print(f"\n{'=' * 50}")
    print(f"Processing subset {i + 1}/{num_subsets}: {subset_name}")
    print(f"Samples: {start_index} to {end_index - 1}")
    print(f"{'=' * 50}")

    # Initialize the model for each subset
    processor = AudioPromptDataset(
      model_name_or_path="pretrained_models/Spark-TTS-0.5B", device="cuda"
    )

    # Select the current subset
    dataset_subset = dataset.select(range(start_index, end_index))

    # Process and tokenize the subset with optimizations
    ljspeech_formatted_dir = process_subset(
      dataset_subset, args.data_dir, subset_name, args.num_workers
    )
    processor.tokenize(
      ljspeech_formatted_dir,
      args.output_dir,
      jsonl_path,
      subset_name,
      args.batch_size,
      args.num_workers or 4,
    )

    print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # Clean up GPU memory after each subset
    del ljspeech_formatted_dir, dataset_subset, processor
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
