"""Provider-agnostic chat-model configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = "ollama:qwen3:4b"


def get_app_model_name() -> str:
    """Return the provider-prefixed model used to answer user questions."""
    return os.getenv("PAPER_QA_MODEL", DEFAULT_MODEL)


def get_judge_model_name() -> str:
    """Return the evaluation model, defaulting to the application model."""
    return os.getenv("PAPER_QA_JUDGE_MODEL", get_app_model_name())


def create_chat_model(
    model: str | None = None,
    *,
    temperature: float = 0,
    base_url: str | None = None,
) -> Any:
    """Create any LangChain-supported model from a provider:model value."""
    resolved_model = model or get_app_model_name()
    resolved_base_url = base_url or os.getenv("PAPER_QA_MODEL_BASE_URL")
    options: dict[str, Any] = {"temperature": temperature}
    if resolved_base_url:
        options["base_url"] = resolved_base_url
    return init_chat_model(resolved_model, **options)
