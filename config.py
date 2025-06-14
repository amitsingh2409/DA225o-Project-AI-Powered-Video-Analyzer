import os
from pathlib import Path

# Base paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = BASE_DIR / "storage"
TEMP_DIR = STORAGE_DIR / "temp"
UPLOADS_DIR = STORAGE_DIR / "uploads"

# Ensure directories exist
for directory in [STORAGE_DIR, TEMP_DIR, UPLOADS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Database configuration
DB_PATH = str(STORAGE_DIR / "video_data.db")

# Whisper configuration
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

# vLLM configuration
VLLM_MODEL = "Qwen/Qwen3-8B"  # Choose your preferred model
VLLM_PORT = 3000
VLLM_HOST = "0.0.0.0"
VLLM_MAX_MODEL_LEN = 4096

# Vector store configuration
VECTOR_DB_PATH = str(STORAGE_DIR / "vector_store")
