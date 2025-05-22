import json
import re
import secrets
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Union


def normalize_name(name: str) -> str:
  """
  Normalize a name by removing accents, converting to lowercase, and replacing spaces with hyphens.

  Args:
    name (str): The input string to normalize.

  Returns:
    str: The normalized string.
  """
  output = unicodedata.normalize("NFD", name)
  output = "".join(c for c in output if unicodedata.category(c) != "Mn")
  return output.lower().replace(" ", "-")


def convert_duration(time_str: Optional[str], unit: str = "second") -> Optional[float]:
  """
  Convert a time string in the format "HH:MM:SS" or "MM:SS" to the specified unit.

  Args:
    time_str (Optional[str]): The input time string.
    unit (str): The unit to convert to ("second", "minute", or "hour").

  Returns:
    Optional[float]: Duration in the desired unit, or None if parsing fails.
  """

  def clean_duration(raw_duration: str) -> str:
    cleaned = re.sub(r"[;:]+", ":", raw_duration.strip())
    return cleaned

  if time_str is None:
    return None

  time_str = clean_duration(time_str)
  time_values = time_str.split(":")
  try:
    total_seconds, idx = 0, 0
    for num in reversed(time_values):
      if num == "":
        continue
      total_seconds += int(num) * 60**idx
      idx += 1

    unit = unit.lower()
    if unit == "second":
      return total_seconds
    elif unit == "minute":
      return round(total_seconds / 60, 4)
    elif unit == "hour":
      return round(total_seconds / 3600, 4)
    else:
      print(f"Invalid unit: {unit}")
      return None

  except Exception as e:
    print(f"Error converting duration: {e}")
    return None


class Narrator:
  """
  Represents a narrator with optional metadata like age, gender, and voice properties.
  """

  def __init__(
    self,
    name: str,
    id: Optional[str] = None,
    url: Optional[str] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    dialect: Optional[str] = None,
    tone: Optional[str] = None,
    volume: Optional[str] = None,
    speed: Optional[str] = None,
    pitch: Optional[str] = None,
  ):
    self.id = id or self.generate_id()
    self.name = name
    self.dialect = dialect
    self.gender = gender
    self.age = age
    self.url = url
    self.tone = tone
    self.volume = volume
    self.speed = speed
    self.pitch = pitch

  def to_dict(self):
    """Convert the narrator to a dictionary."""
    return self.__dict__

  def __repr__(self):
    return f"Narrator(id={self.id}, name={self.name}, url={self.url})"

  def __str__(self):
    return self.__repr__()

  def __eq__(self, other):
    """Check equality based on ID."""
    if not isinstance(other, Narrator):
      return False
    return self.id == other.id

  @classmethod
  def from_json(cls, json_path: Union[str, Path]):
    """
    Load a narrator from a JSON file.

    Args:
      json_path (str | Path): Path to the JSON file.

    Returns:
      Narrator: Loaded narrator object.
    """
    with open(str(json_path), encoding="utf-8") as f:
      data = json.load(f)
    return cls(**data)

  @classmethod
  def from_dict(cls, data: Dict):
    """
    Load a narrator from a dictionary.

    Args:
      data (Dict): Dictionary of narrator fields.

    Returns:
      Narrator: Loaded narrator object.
    """
    return cls(**data)

  @staticmethod
  def generate_id(length: int = 8):
    """
    Generate a random hexadecimal ID.

    Args:
      length (int): Desired length of the ID.

    Returns:
      str: Generated ID.
    """
    return secrets.token_hex(length // 2)

  @classmethod
  def from_csv_row(cls, row: Dict[str, str]) -> "Narrator":
    """
    Load a narrator from a CSV row.

    Args:
      row (Dict[str, str]): Dictionary from a CSV row.

    Returns:
      Narrator: Loaded narrator object.
    """
    try:
      narrator_id = row.get("narrator_id", "")
    except (ValueError, TypeError):
      narrator_id = None

    return cls(
      id=narrator_id or cls.generate_id(),
      name=row.get("name", ""),
      url=row.get("url"),
      age=int(row["age"]) if row.get("age") and row["age"].isdigit() else None,
      gender=row.get("gender"),
      dialect=row.get("dialect"),
      tone=row.get("tone"),
      volume=row.get("volume"),
      speed=row.get("speed"),
      pitch=row.get("pitch"),
    )


class Book:
  """
  Represents a book with metadata such as name, author, narrator(s), and file paths.
  """

  def __init__(
    self,
    name: str,
    narrator: Union[Narrator, List[Narrator]],
    duration: Union[str, float],
    text_path: Optional[Union[str, Path]] = None,
    audio_path: Optional[Union[str, Path]] = None,
    alignment_path: Optional[Union[str, Path]] = None,
    id: Optional[str] = None,
    author: Optional[str] = None,
    text_url: Optional[str] = None,
    audio_url: Optional[str] = None,
  ):
    """
    Initialize a Book object.

    Args:
      name (str): Name of the book.
      narrator (Narrator | List[Narrator]): Narrator(s) for the book.
      duration (str): Duration string (e.g. "01:20:00").
      text_path (str | Path, optional): Local path to text file.
      audio_path (str | Path, optional): Local path to audio file.
      id (str, optional): Optional unique ID.
      author (str, optional): Author's name.
      text_url (str, optional): URL to online text.
      audio_url (str, optional): URL to online audio.
    """
    
    duration_hour = convert_duration(duration, unit="hour") if isinstance(duration, str) else duration

    self.id = id or self.generate_id()
    self.name = normalize_name(name)
    self.text_path = text_path
    self.audio_path = audio_path
    self.narrator = narrator
    self.duration = duration_hour
    self.author = author
    self.text_url = text_url
    self.audio_url = audio_url
    self.alignment_path = alignment_path

  @classmethod
  def from_json(cls, json_path: Union[str, Path]) -> "Book":
    """
    Load a Book object from a JSON file.

    Args:
      json_path (str | Path): Path to the JSON file.

    Returns:
      Book: Loaded book object.
    """
    with open(json_path, encoding="utf-8") as f:
      data = json.load(f)

    narrators_data = data.get("narrator", [])
    if isinstance(narrators_data, dict):
      narrators = [Narrator(**narrators_data)]
    else:
      narrators = [Narrator(**n) for n in narrators_data]

    return cls(
      name=data.get("name"),
      text_url=data.get("text_url"),
      audio_url=data.get("audio_url"),
      author=data.get("author"),
      duration=data.get("duration"),
      narrator=narrators,
    )

  def to_dict(self):
    """
    Convert the Book object to a dictionary.

    Returns:
      dict: Dictionary representation of the book.
    """
    result = self.__dict__.copy()
    if self.narrator:
      if isinstance(self.narrator, list):
        result["narrator"] = [n.to_dict() for n in self.narrator]
      else:
        result["narrator"] = self.narrator.to_dict()
    return result

  def save_json(self, path: Union[str, Path]):
    """
    Save the Book object to a JSON file.

    Args:
      path (str | Path): Output path for the JSON file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

  def __repr__(self):
    return (
      f"Book("
      f"id={self.id}, "
      f"name={self.name}, "
      f"author={self.author}, "
      f"narrator={self.narrator}, "
      f"duration={self.duration}, "
      f"text_url={self.text_url}, "
      f"audio_url={self.audio_url}"
      f")"
    )

  @staticmethod
  def generate_id(length: int = 8):
    """
    Generate a random hexadecimal ID for the book.

    Args:
      length (int): Length of the ID.

    Returns:
      str: Randomly generated ID.
    """
    return secrets.token_hex(length // 2)

  def update_paths(
    self, 
    text_path: Optional[Union[str, Path]] = None, 
    audio_path: Optional[Union[str, Path]] = None, 
    alignment_path: Optional[Union[str, Path]] = None,
  ):
    """
    Update the local file paths for the book.

    Args:
      text_path (str | Path): Local path to the text file.
      audio_path (str | Path): Local path to the audio file.
    """
    if text_path is not None:
      self.text_path = text_path
    if audio_path is not None:
      self.audio_path = audio_path
    if alignment_path is not None:
      self.alignment_path = alignment_path
