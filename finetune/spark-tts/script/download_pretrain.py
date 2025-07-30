from huggingface_hub import snapshot_download

snapshot_download(
  "DragonLineageAI/Vi-SparkTTS-0.5B", local_dir="pretrained_models/Spark-TTS-0.5B"
)
