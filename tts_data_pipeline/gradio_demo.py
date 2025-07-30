# app.py

# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "gradio",
#   "numpy",
#   "soundfile",
#   "torchaudio",
#   "accelerate",
# ]
# ///

import os
import re

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer

# --- Global Configurations ---
MODEL_ID = "qhuy242/spark-tts-finetuned-on-vnavc-dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_AUDIO_TOKENS = 2048  # Max tokens for audio part
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 1.0  # Changed to float for consistency


# --- Model and Tokenizer Loading ---
tokenizer = None
model = None
audio_tokenizer = None
sample_rate = 16000  # Default, will be updated from audio_tokenizer config


def load_model_and_tokenizers():
  """Loads the pre-trained model, tokenizer, and audio processor."""
  global tokenizer, model, audio_tokenizer, sample_rate

  print(f"Using device: {DEVICE}")
  print(f"Loading model and tokenizer for {MODEL_ID}...")

  try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(DEVICE)
    audio_tokenizer = AutoProcessor.from_pretrained(MODEL_ID)

    # Enable native 2x faster inference
    if hasattr(model, "for_inference"):
      model.for_inference()
    else:
      print(
        "Warning: 'for_inference' method not found on the model. Skipping native inference speedup."
      )

    sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
    print("Model and tokenizers loaded successfully.")
  except Exception as e:
    print(f"Error loading model components: {e}")
    # Optionally re-raise or handle more gracefully for a Gradio app
    raise RuntimeError(f"Failed to load model components: {e}")


# --- Speech Generation Core Logic ---
def generate_speech_from_text(
  text: str,
  temperature: float = DEFAULT_TEMPERATURE,
  top_k: int = DEFAULT_TOP_K,
  top_p: float = DEFAULT_TOP_P,
) -> np.ndarray:
  """
  Generates speech audio from text using the loaded model.

  Args:
      text (str): The text input to be converted to speech.
      temperature (float): Sampling temperature for generation.
      top_k (int): Top-k sampling parameter.
      top_p (float): Top-p (nucleus) sampling parameter.

  Returns:
      np.ndarray: Generated waveform as a NumPy array.
  """
  if not text:
    print("Input text is empty. Returning empty audio.")
    return np.array([], dtype=np.float32)

  if model is None or tokenizer is None or audio_tokenizer is None:
    print(
      "Model components not loaded. Please ensure load_model_and_tokenizers() was called."
    )
    raise RuntimeError("Model components are not loaded.")

  prompt = "".join(
    [
      "<|task_tts|>",
      "<|start_content|>",
      text,
      "<|end_content|>",
      "<|start_global_token|>",
    ]
  )

  model_inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)

  print("Generating token sequence...")
  generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=MAX_NEW_AUDIO_TOKENS,
    do_sample=True,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
  )
  print("Token sequence generated.")

  generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1] :]
  predicts_text = tokenizer.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False
  )[0]

  semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
  if not semantic_matches:
    print("Warning: No semantic tokens found in the generated output.")
    return np.array([], dtype=np.float32)

  pred_semantic_ids = (
    torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
  )

  global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
  if not global_matches:
    print(
      "Warning: No global tokens found in the generated output (controllable mode). Using default zero global token."
    )
    pred_global_ids = torch.zeros((1, 1), dtype=torch.long)
  else:
    pred_global_ids = (
      torch.tensor([int(token) for token in global_matches]).long().unsqueeze(0)
    )

  pred_global_ids = pred_global_ids.unsqueeze(0)  # Ensure (1, 1, N_global)

  print(f"Found {pred_semantic_ids.shape[1]} semantic tokens.")
  print(f"Found {pred_global_ids.shape[2]} global tokens.")

  print("Detokenizing audio tokens...")
  # Ensure audio_tokenizer and its internal model are on the correct device
  audio_tokenizer.device = DEVICE  # Set device explicitly for the processor
  audio_tokenizer.model.to(DEVICE)  # Move the internal model to device
  with torch.no_grad():
    wav_np = audio_tokenizer.detokenize(
      pred_global_ids.to(DEVICE).squeeze(0),  # Shape (1, N_global)
      pred_semantic_ids.to(DEVICE),  # Shape (1, N_semantic)
    )
  print("Detokenization complete.")

  return wav_np


# --- Gradio Interface Function ---
def gradio_interface_fn(
  text_input: str, temperature: float, top_k: int, top_p: float
) -> tuple[int, np.ndarray]:
  """
  Wrapper function for Gradio interface to generate speech.
  Takes text input and generation parameters, returns sample rate and audio array.
  """
  try:
    generated_waveform = generate_speech_from_text(
      text_input, temperature=temperature, top_k=top_k, top_p=top_p
    )
    if generated_waveform.size > 0:
      return sample_rate, generated_waveform
    else:
      gr.Warning(
        "Không thể tạo âm thanh. Vui lòng thử lại với văn bản khác hoặc điều chỉnh tham số."
      )
      return sample_rate, np.array([], dtype=np.float32)
  except Exception as e:
    gr.Error(f"Đã xảy ra lỗi trong quá trình tạo âm thanh: {e}")
    return sample_rate, np.array([], dtype=np.float32)


# --- Gradio UI Definition ---
def create_gradio_interface():
  """Creates and returns the Gradio Interface object."""
  return gr.Interface(
    fn=gradio_interface_fn,
    inputs=[
      gr.Textbox(
        lines=3,
        placeholder="Nhập văn bản tiếng Việt để chuyển thành giọng nói...",
        label="Văn bản đầu vào",
      ),
      gr.Slider(
        minimum=0.1,
        maximum=2.0,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
        label="Nhiệt độ (Temperature)",
        info="Kiểm soát tính ngẫu nhiên của giọng nói. Giá trị cao hơn tạo ra giọng nói đa dạng hơn.",
      ),
      gr.Slider(
        minimum=1,
        maximum=250,  # A reasonable max for top_k
        value=DEFAULT_TOP_K,
        step=1,
        label="Top-K",
        info="Chỉ xem xét K từ có khả năng cao nhất.",
      ),
      gr.Slider(
        minimum=0.1,
        maximum=1.0,
        value=DEFAULT_TOP_P,
        step=0.05,
        label="Top-P (Nucleus Sampling)",
        info="Chỉ xem xét các từ mà tổng xác suất của chúng đạt P.",
      ),
    ],
    outputs=gr.Audio(
      label="Âm thanh được tạo",
      type="numpy",
      autoplay=True,
    ),
    title="🌟 SparkTTS Tiếng Việt: Chuyển Văn Bản Thành Giọng Nói 🌟",
    description=(
      "Sử dụng mô hình SparkTTS đã tinh chỉnh trên tập dữ liệu VNAVC để tạo giọng nói từ văn bản tiếng Việt. "
      "Bạn có thể điều chỉnh các tham số tạo để kiểm soát kết quả."
    ),
    examples=[
      [
        "Xin chào, tôi là mô hình chuyển đổi văn bản thành giọng nói.",
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_K,
        DEFAULT_TOP_P,
      ],
      [
        "Cảm ơn bạn đã lắng nghe. Hy vọng bạn thích demo này.",
        0.6,
        50,
        0.95,
      ],  # Ví dụ với tham số khác
      [
        "Hồ Chí Minh là thành phố lớn nhất Việt Nam.",
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_K,
        DEFAULT_TOP_P,
      ],
    ],
    allow_flagging="never",
    css="footer {visibility: hidden}",  # Hide Gradio footer for a cleaner look
  )


# --- Main Execution Block ---
if __name__ == "__main__":
  load_model_and_tokenizers()
  iface = create_gradio_interface()
  print("Launching Gradio interface...")
  iface.queue().launch(share=True)
