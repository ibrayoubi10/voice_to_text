from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = BASE_DIR / "recordings"
OUTPUTS_DIR = BASE_DIR / "outputs"

SAMPLE_RATE = 16000
CHANNELS = 1

WHISPER_MODEL_SIZE = "base"   # tiny, base, small, medium
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "llama3"

DEFAULT_LANGUAGE = "fr"
APP_TITLE = "Voice to Text + Structured Summary"
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 820