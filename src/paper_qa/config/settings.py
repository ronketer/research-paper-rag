from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ENV_FILE = PROJECT_ROOT / ".env"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"

DEFAULT_CHAT_MODEL = "ollama:qwen3:4b"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
