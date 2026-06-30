# Explicit configuration — edit these before running
ESCRIPTORIUM_URL="http://localhost:8080"
ESCRIPTORIUM_USERNAME="admin"
ESCRIPTORIUM_PASSWORD="admin"
DOCUMENT_ID=6
TRANSCRIPTION_NAME="manual"
MODEL_PATH="catmus-print-fondue-large.mlmodel"
EPOCHS=1
BATCH_SIZE=2
OUTPUT="catmus-print-fondue-ft.safetensors"

uv run python -m training.cli \
    --url "$ESCRIPTORIUM_URL" \
    --username "$ESCRIPTORIUM_USERNAME" \
    --password "$ESCRIPTORIUM_PASSWORD" \
    --document-id "$DOCUMENT_ID" \
    --transcription-name "$TRANSCRIPTION_NAME" \
    --model-path "$MODEL_PATH" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output "$OUTPUT"
