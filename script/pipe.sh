#!/bin/bash

set -e

# ===============================
# Configurable variables
# ===============================
BOOK_LIST_FILE="$1"
PROJECT_ROOT="$(pwd)"
RAW_AUDIO_DIR="$PROJECT_ROOT/data/audio/raw"
QUALIFIED_AUDIO_DIR="$PROJECT_ROOT/data/audio/qualified"
TEXT_PDF_DIR="$PROJECT_ROOT/data/text/pdf"
TEXT_SENTENCE_DIR="$PROJECT_ROOT/data/text/sentence"
JOBS=4

if [[ -z "$BOOK_LIST_FILE" ]]; then
  echo "❌  Missing argument: path to the TXT file containing book names."
  echo "👉 Usage: ./run_pipeline.sh books.txt"
  exit 1
fi


# ===============================
# Process each book
# ===============================
while IFS= read -r BOOK_NAME || [[ -n "$BOOK_NAME" ]]; do
  echo "=============================="
  echo "📚 Processing book: $BOOK_NAME"
  echo "=============================="

  # Step 1: Crawler
  echo "🔍 Downloading audiobook: $BOOK_NAME"
  python3 "$PROJECT_ROOT/tts_data_pipeline/crawler/main.py" \
  --download query \
  --name "$BOOK_NAME"

  # Step 2: Pre-processing text
  PDF_PATH="$TEXT_PDF_DIR/$BOOK_NAME.pdf"
  echo "🧹 Pre-processing text: $PDF_PATH"
  python3 -m tts_data_pipeline.pre_processing.main --target text --path "$PDF_PATH"

  # Step 2: Pre-processing audio
  RAW_AUDIO_PATH="$RAW_AUDIO_DIR/$BOOK_NAME"
  echo "🧹 Pre-processing audio: $RAW_AUDIO_PATH"
  python3 -m tts_data_pipeline.pre_processing.main --target audio --path "$RAW_AUDIO_PATH"

  # Step 3: Alignment
  FINAL_AUDIO_PATH="$QUALIFIED_AUDIO_DIR/$BOOK_NAME.wav"
  FINAL_TEXT_PATH="$TEXT_SENTENCE_DIR/$BOOK_NAME.txt"
  echo "🪢 Aligning audio and text for: $BOOK_NAME"
  python3 -m tts_data_pipeline.alignment.main \
  --audio "$FINAL_AUDIO_PATH" \
  --text "$FINAL_TEXT_PATH" \
  --split \
  --jobs "$JOBS"

  # Step 4: Cleanup
  echo "🧹 Cleaning up to save disk space..."
  # rm -rf "$RAW_AUDIO_PATH"                     # Delete raw audio folder
  # rm -rf "$TEXT_PDF_DIR"                       # Delete PDF text folder
  # rm -f "$FINAL_AUDIO_PATH"                    # Delete processed audio
  # rm -f "$FINAL_TEXT_PATH"                     # Delete processed text

  echo "✅ Finished processing $BOOK_NAME"

done < "$BOOK_LIST_FILE"

echo "🏁 All books have been processed successfully!"
