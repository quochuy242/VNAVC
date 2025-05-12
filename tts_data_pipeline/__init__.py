from dataclasses import dataclass


@dataclass
class BookInfo:
  name: str
  audio_path: str
  text_path: str
  audio_url: str
  text_url: str
  narrator_id: int
  narrator_name: str
  origin_sample_rate: int
