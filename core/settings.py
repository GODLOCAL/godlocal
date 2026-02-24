"""core/settings.py — БОГ || OASIS v6 unified config via pydantic-settings + .env"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    model: str = "qwen3:8b"
    device: str = "mps"                   # mps | cuda | cpu

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""                      # GODLOCAL_API_KEY — empty = no auth

    # Sleep cycle
    sleep_hour: int = 1
    sleep_minute: int = 0

    # Memory
    short_term_limit: int = 50
    memory_path: str = "godlocal_data/memory"
    data_path: str = "godlocal_data"
    soul_file: str = "BOH_OASIS.md"

    # Docker sandbox
    sandbox_image: str = "godlocal-sandbox:latest"
    sandbox_timeout: int = 60              # seconds

    # Sonic gateway
    sonic_host: str = "0.0.0.0"
    sonic_port: int = 9000

    # AutoGenesis
    autogenesis_apply: bool = False        # set True to enable live code patching
    autogenesis_max_revisions: int = 2

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GODLOCAL_",
        extra="ignore",
    )


settings = Settings()
