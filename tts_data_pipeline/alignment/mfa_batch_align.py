import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from rich.console import Console

app = typer.Typer(help="MFA Batch Alignment Tool")
console = Console()


def run_cmd(cmd: List[str], verbose: bool = False) -> None:
  """
  Run a command in the shell and print output to console.
  """
  if verbose:
    console.print(f"[bold green]Running command:[/bold green] {' '.join(cmd)}")

  try:
    result = subprocess.run(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      check=True,
    )
    if verbose:
      console.print("[bold green]Command executed successfully.[/bold green]")
      console.print("[bold blue]STDOUT:[/bold blue]")
      console.print(result.stdout)
      console.print("[bold yellow]STDERR:[/bold yellow]")
      console.print(result.stderr)
  except subprocess.CalledProcessError as e:
    console.print(f"[bold red]Error executing command:[/bold red] {e}")
    console.print("[bold red]STDERR:[/bold red]")
    console.print(e.stderr)


@app.command()
def setup_corpus(
  corpus_path: Path = typer.Argument(
    ...,
    help="Path to the corpus directory.",
    default_factory=lambda: Path("./data/mfa_corpus"),
  ),
  text_path: Path = typer.Argument(
    ...,
    help="Path to the processed text files directory.",
    default_factory=lambda: Path("./data/text/processed"),
  ),
  audio_path: Path = typer.Argument(
    ...,
    help="Path to the processed audio files directory.",
    default_factory=lambda: Path("./data/audio/processed"),
  ),
) -> List[Tuple[Path, Path]]:
  """
  Ensure the corpus directory exists and is set up correctly. The conditions are:
  - The corpus directory must exist.
  - The text directory must exist and contain processed text files.
  - The audio directory must exist and contain processed audio files.
  - All of text files must have corresponding audio files.

  Args:
    corpus_path (Path): Path to the corpus directory.
    text_path (Path): Path to the processed text files directory.
    audio_path (Path): Path to the processed audio files directory.
  """

  # Check if paths exist
  if not corpus_path.exists():
    console.print(
      f"[bold red]Error:[/bold red] Corpus path does not exist: {corpus_path}, then create it."
    )
    corpus_path.mkdir(parents=True, exist_ok=True)
  if not text_path.exists():
    console.print(f"[bold red]Error:[/bold red] Text path does not exist: {text_path}")
    raise FileNotFoundError(f"Text path does not exist: {text_path}")
  if not audio_path.exists():
    console.print(
      f"[bold red]Error:[/bold red] Audio path does not exist: {audio_path}"
    )
    raise FileNotFoundError(f"Audio path does not exist: {audio_path}")

  text_files = list(text_path.glob("*.txt"))
  audio_files = list(audio_path.glob("*.wav"))

  if len(text_files) != len(audio_files):
    console.print(
      "[bold orange]Warning:[/bold orange] The number of text files does not match the number of audio files."
    )

    # Paring text and audio files
    audio_text_pairs = []
    idx_text, idx_audio = 0, 0
    while idx_text < len(text_files) and idx_audio < len(audio_files):
      if text_files[idx_text].stem < audio_files[idx_audio].stem:
        idx_text += 1
      elif text_files[idx_text].stem > audio_files[idx_audio].stem:
        idx_audio += 1
      else:
        audio_text_pairs.append((text_files[idx_text], audio_files[idx_audio]))
        idx_text += 1
        idx_audio += 1

    console.print(
      f"[bold blue]Info:[/bold blue] The number of text files matches the number of audio files {len(audio_text_pairs)}."
    )
  else:
    audio_text_pairs = list(zip(sorted(text_files), sorted(audio_files)))
    console.print(
      f"[bold green]Info:[/bold green] The number of text files matches the number of audio files: {len(audio_text_pairs)}."
    )

  # Check if there are any audio-text pairs
  if not audio_text_pairs:
    console.print(
      "[bold red]Error:[/bold red] No matching text and audio files found. Please check your corpus."
    )
    raise ValueError("No matching text and audio files found.")

  # Moving all text and audio files to the corpus directory
  for text_file, audio_file in audio_text_pairs:
    new_text_path = corpus_path / "text" / text_file.name
    new_audio_path = corpus_path / "audio" / audio_file.name

    if not new_text_path.parent.exists():
      new_text_path.parent.mkdir(parents=True, exist_ok=True)
    if not new_audio_path.parent.exists():
      new_audio_path.parent.mkdir(parents=True, exist_ok=True)

    text_file.rename(new_text_path)
    audio_file.rename(new_audio_path)

  console.print(
    f"[bold green]Info:[/bold green] Corpus setup completed with {len(audio_text_pairs)} pairs."
  )
  return audio_text_pairs


@app.command()
def check_mfa(
  model_name: Optional[str] = typer.Argument(
    None, help="Name of the MFA model to check."
  ),
):
  """
  Check if the MFA (Montreal Forced Aligner) is installed and if the specified model exists.
  """
  if model_name is None:
    console.print(
      "[bold red]Error:[/bold red] Model name is None, use [green]vietnamese_mfa[/green] as default model instead."
    )
    model_name = "vietnamese_mfa"

  mfa_cmd_version = ["mfa", "--version"]
  version_output = subprocess.run(
    mfa_cmd_version,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
  )
  if version_output.returncode != 0:
    console.print(
      "[bold red]Error:[/bold red] MFA is not installed or not found in PATH."
    )
    return

  mfa_cmd_dict = ["mfa", "model", "list", "dictionary"]
  mfa_cmd_acoustic = ["mfa", "model", "list", "acoustic"]

  dict_output = subprocess.run(
    mfa_cmd_dict,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
  )
  acoustic_output = subprocess.run(
    mfa_cmd_acoustic,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
  )

  if model_name not in dict_output.stdout or model_name not in acoustic_output.stdout:
    console.print(
      f"[bold red]Error:[/bold red] Model '{model_name}' does not exist in MFA dictionaries or acoustic models."
    )
    console.print(
      "[bold red]Please ensure the model is installed correctly.[/bold red]. \nCheck README_MFA.md for more details."
    )
    return

  console.print(
    f"[bold green]MFA is installed and model '{model_name}' exists.[/bold green]"
  )


@app.command()
def validate_corpus(
  corpus_path: str = typer.Argument(..., help="Path to the corpus directory."),
  dict_path: Optional[str] = typer.Argument(
    None, help="Path to the dictionary file (optional)."
  ),
  acoustic_model: Optional[str] = typer.Argument(
    None, help="Name of the MFA model to use for validation."
  ),
  jobs: int = typer.Option(4, "--jobs", "-j", help="Number of parallel jobs to run."),
  verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output."),
):
  """
  Validate the corpus directory using MFA.
  """
  if acoustic_model is None:
    acoustic_model = "vietnamese_mfa"  # Default model

  if dict_path is None:
    dict_path = f"{acoustic_model}"  # Default dictionary path based on model name

  cmd = [
    "mfa",
    "validate",
    "-j",
    str(jobs),
    corpus_path,
    dict_path,
  ]

  run_cmd(cmd, verbose)


@app.command()
def align(
  corpus_path: str = typer.Argument(..., help="Path to the corpus directory."),
  dict_path: Optional[str] = typer.Argument(None, help="Path to the dictionary file."),
  acoustic_model: Optional[str] = typer.Argument(
    None, help="Name of the MFA model to use for alignment."
  ),
  clean: bool = typer.Option(
    False, "--clean", "-c", help="Clean the output directory before alignment."
  ),
  output_dir: Optional[str] = typer.Option(
    None, "--output-dir", "-o", help="Path to the output directory."
  ),
  jobs: int = typer.Option(4, "--jobs", "-j", help="Number of parallel jobs to run."),
  verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output."),
):
  """
  Align the corpus using MFA.
  """
  if acoustic_model is None:
    acoustic_model = "vietnamese_mfa"  # Default model

  if dict_path is None:
    dict_path = f"{acoustic_model}"  # Default dictionary path based on model name

  if output_dir is None:
    output_dir = Path("mfa_output")
    output_dir.mkdir(parents=True, exist_ok=True)

  cmd = [
    "mfa",
    "align",
  ]

  if clean:
    cmd.append("--clean")

  cmd.extend(
    [
      "-j",
      str(jobs),
      corpus_path,
      dict_path,
      acoustic_model,
      str(output_dir),
    ]
  )

  run_cmd(cmd, verbose)
