"""tests/test_settings.py — core/settings.py coverage"""
import os
import pytest
from unittest.mock import patch


def test_settings_defaults():
    from core.settings import Settings
    s = Settings()
    assert s.model == "qwen3:8b"
    assert s.device == "mps"
    assert s.api_port == 8000
    assert s.short_term_limit == 50
    assert s.autogenesis_apply is False


def test_settings_env_override():
    with patch.dict(os.environ, {
        "GODLOCAL_MODEL": "llama3:8b",
        "GODLOCAL_API_PORT": "9999",
        "GODLOCAL_AUTOGENESIS_APPLY": "true",
    }):
        from core.settings import Settings
        s = Settings()
        assert s.model == "llama3:8b"
        assert s.api_port == 9999
        assert s.autogenesis_apply is True


def test_settings_api_key_empty_by_default():
    from core.settings import Settings
    s = Settings()
    assert s.api_key == ""


def test_settings_memory_path():
    from core.settings import Settings
    s = Settings()
    assert "memory" in s.memory_path
