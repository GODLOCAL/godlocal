"""
tests/test_utils.py — Basic tests for shared utilities
"""
import os
import tempfile
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import detect_device, Capabilities, format_status, atomic_write


def test_detect_device_returns_string():
    device = detect_device()
    assert device in ("cuda", "rocm", "mps", "cpu"), f"Unexpected device: {device}"


def test_capabilities_has_required_attrs():
    assert hasattr(Capabilities, "ollama")
    assert hasattr(Capabilities, "chroma")
    assert hasattr(Capabilities, "self_evolve")
    assert hasattr(Capabilities, "paroquant")


def test_capabilities_summary_has_device():
    summary = Capabilities.summary()
    assert "device" in summary
    assert summary["device"] in ("cuda", "rocm", "mps", "cpu")


def test_format_status_basic():
    data = {
        "version": "v5",
        "soul_loaded": True,
        "memory_entries": 42,
        "llm_engine": "ollama",
        "device": "cpu",
        "self_evolve": True,
        "uptime": "1h 23m",
    }
    result = format_status(data)
    assert "GodLocal" in result
    assert "ollama" in result
    assert "cpu" in result


def test_atomic_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.txt")
        atomic_write(path, "hello world")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == "hello world"


def test_atomic_write_overwrites():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.txt")
        atomic_write(path, "first")
        atomic_write(path, "second")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == "second"
