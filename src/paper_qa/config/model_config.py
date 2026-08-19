import os
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from paper_qa.config.settings import (
    DEFAULT_CHAT_MODEL,
    ENV_FILE,
)

load_dotenv(ENV_FILE)


def get_app_model_name() -> str:
    """Return the provider-prefixed configured chat model."""

    return os.getenv(
        "PAPER_QA_MODEL",
        DEFAULT_CHAT_MODEL,
    )


def create_chat_model(
    model: str | None = None,
    *,
    temperature: float = 0,
    base_url: str | None = None,
) -> Any:
    """Create the configured LangChain chat model."""

    resolved_model = (
        model
        or get_app_model_name()
    )

    resolved_base_url = (
        base_url
        or os.getenv("PAPER_QA_MODEL_BASE_URL")
    )

    options: dict[str, Any] = {
        "temperature": temperature,
    }

    if resolved_base_url:
        options["base_url"] = resolved_base_url

    return init_chat_model(
        resolved_model,
        **options,
    )
