from tts_data_pipeline.alignment.utils import align_audio_text

align_audio_text(
  audio_path="data/audio/qualified/mat-biec/mat-biec_1.wav",
  text_path="data/text/sentences/mat-biec.txt",
  output_path="data/alignment/test.tsv",
)
