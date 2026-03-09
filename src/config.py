"""
Centralized application configuration using Pydantic Settings.

All values can be overridden via environment variables or a `.env` file.

Optional API keys (leave unset to disable the corresponding data source):
  NEWSAPI_KEY    — NewsAPI.org free tier  (https://newsapi.org/register)
  HIBP_API_KEY   — Have I Been Pwned v3   (https://haveibeenpwned.com/API/Key)
"""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration — required fields have no defaults and must be provided."""

    # --- Data Storage Root ---
    # All investigation data is stored under: {data_dir}/{investigation_id}/
    data_dir: str = "./data"

    # --- Optional API Keys ---
    newsapi_key: Optional[str] = None  # Set NEWSAPI_KEY in .env to enable NewsAPI
    hibp_api_key: Optional[str] = None  # Set HIBP_API_KEY in .env to enable HIBP
    firecrawl_api_key: Optional[str] = (
        None  # Set FIRECRAWL_API_KEY in .env to enable Firecrawl API for deep web scraping
    )

    # --- Sizing Limits ---
    max_search_results: int = 5
    scrape_content_chars_max: int = 3000
    planner_query_count: int = 5

    # --- LLM Settings ---
    ollama_model: str = "qwen3:4b-instruct"
    embedding_model: str = "qwen3-embedding:0.6b"
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2"
    nli_model: str = "tasksource/ModernBERT-base-nli"
    planner_temperature: float = 0.2
    gatherer_temperature: float = 0.1
    synthesis_temperature: float = 0.1
    briefing_temperature: float = 0.2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


config = Settings()
