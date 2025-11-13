"""
Configuration management for the AI Threat Detection Agent.

Loads settings from environment variables with sensible defaults.
Supports both mock mode (for offline demos) and real LLM API mode.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # LLM API Configuration
    hf_api_key: Optional[str] = Field(
        default=None,
        description="Hugging Face API key for inference API access"
    )

    hf_api_url: str = Field(
        default="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        description="Hugging Face model endpoint URL"
    )

    # Mode Selection
    use_mock_model: bool = Field(
        default=True,
        description="Use mock LLM responses (deterministic, no API calls)"
    )

    # Storage Configuration
    storage_path: str = Field(
        default="data/predictions.csv",
        description="Path to store prediction history"
    )

    # LLM Request Configuration
    llm_timeout: int = Field(
        default=30,
        description="Timeout for LLM API requests in seconds"
    )

    llm_max_tokens: int = Field(
        default=200,
        description="Maximum tokens for LLM response"
    )

    llm_temperature: float = Field(
        default=0.3,
        description="Temperature for LLM sampling (lower = more deterministic)"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def is_mock_mode() -> bool:
    """Check if the application is running in mock mode."""
    return settings.use_mock_model


def has_api_key() -> bool:
    """Check if a valid API key is configured."""
    return settings.hf_api_key is not None and len(settings.hf_api_key) > 0


def validate_config() -> tuple[bool, list[str]]:
    """
    Validate configuration and return status + warnings.

    Returns:
        (is_valid, warnings): Tuple of validation status and list of warning messages
    """
    warnings = []

    if not is_mock_mode():
        if not has_api_key():
            warnings.append(
                "⚠️  Real LLM mode enabled but no API key found. "
                "Set HF_API_KEY in .env or enable USE_MOCK_MODEL=true"
            )

        if settings.hf_api_url == "":
            warnings.append("⚠️  HF_API_URL is empty")

    # Storage path check
    storage_dir = os.path.dirname(settings.storage_path)
    if storage_dir and not os.path.exists(storage_dir):
        try:
            os.makedirs(storage_dir, exist_ok=True)
        except Exception as e:
            warnings.append(f"⚠️  Could not create storage directory: {e}")

    is_valid = len(warnings) == 0 or is_mock_mode()

    return is_valid, warnings
