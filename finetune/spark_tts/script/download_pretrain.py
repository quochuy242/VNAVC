import argparse
from huggingface_hub import snapshot_download
import os


def download_spark_tts(version="vi"):
  """
  Downloads the specified version of the Spark-TTS model from Hugging Face Hub.

  Args:
      version (str): The version to download. Can be 'vi' for Vietnamese
                     or 'original' for the original version.
  """
  if version == "vi":
    repo_id = "DragonLineageAI/Vi-SparkTTS-0.5B"
    local_dir = "pretrained_models/Spark-TTS-0.5B"
    print(f"Downloading Vietnamese version of Spark-TTS from {repo_id}...")
  elif version == "original":
    repo_id = "SparkAudio/Spark-TTS-0.5B"
    local_dir = "pretrained_models/Spark-TTS-0.5B"
    print(f"Downloading original version of Spark-TTS from {repo_id}...")
  else:
    print(f"Error: Invalid version '{version}'. Please choose 'vi' or 'original'.")
    return

  # Create the local directory if it doesn't exist
  os.makedirs(local_dir, exist_ok=True)

  try:
    # Use snapshot_download to download the model
    downloaded_path = snapshot_download(repo_id, local_dir=local_dir)
    print(f"Download complete! Model saved to: {downloaded_path}")
  except Exception as e:
    print(f"An error occurred during download: {e}")


if __name__ == "__main__":
  # Create the parser
  parser = argparse.ArgumentParser(
    description="Download different versions of Spark-TTS model from Hugging Face Hub."
  )

  # Add the argument for the version choice
  parser.add_argument(
    "--version",
    type=str,
    default="vi",
    choices=["vi", "original"],
    help="Specify the model version to download: 'vi' for Vietnamese or 'original' for the base model.",
  )

  # Parse the arguments
  args = parser.parse_args()

  # Call the download function with the chosen version
  download_spark_tts(version=args.version)
