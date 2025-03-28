import torch


# Load Silero model
SILERO_MODEL, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_vad",
    force_reload=True
)
(get_speech_timestamps, _, read_audio, _, _) = utils

