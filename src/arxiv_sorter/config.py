"""Configuration loaded from environment variables."""

import os
from pathlib import Path

# API Configuration
API_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.doubleword.ai/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")

# Batch settings
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1000"))
BATCH_WINDOW_SECONDS = float(os.environ.get("BATCH_WINDOW_SECONDS", "2.0"))
POLL_INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "2.0"))

# Data path - default to arxiv-metadata.parquet in workspace root
_default_parquet = Path(__file__).parent.parent.parent.parent / "arxiv-metadata.parquet"
PARQUET_PATH = os.environ.get("ARXIV_PARQUET_PATH", str(_default_parquet))

# LLM settings
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
