"""
utils.py — GodLocal v5 shared utilities
Centralises: device detection, capability flags, status formatting, atomic writes
"""
import os
import logging

logger = logging.getLogger(__name__)


# ─── Device Detection ────────────────────────────────────────────────────────

def detect_device() -> str:
    """
    Detect the best available compute device.
    Returns: 'cuda', 'rocm', 'mps', or 'cpu'
    """
    try:
        import torch
        if torch.cuda.is_available():
            if hasattr(torch.version, 'hip') and torch.version.hip:
                return 'rocm'
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


# ─── Capability Flags (resolved once at import) ──────────────────────────────

def _check_cap(package: str) -> bool:
    try:
        __import__(package)
        return True
    except ImportError:
        return False


class Capabilities:
    """Single source of truth for optional-dependency availability."""
    ollama: bool = _check_cap('ollama')
    airllm: bool = _check_cap('airllm')
    chroma: bool = _check_cap('chromadb')
    self_evolve: bool = os.path.exists(
        os.path.join(os.path.dirname(__file__), 'self_evolve.py')
    )
    paroquant: bool = _check_cap('transformers') and _check_cap('torch')
    polymarket: bool = os.path.exists(
        os.path.join(os.path.dirname(__file__), 'extensions', 'xzero', 'polymarket_connector.py')
    )

    @classmethod
    def summary(cls) -> dict:
        return {
            'ollama': cls.ollama,
            'airllm': cls.airllm,
            'chroma': cls.chroma,
            'self_evolve': cls.self_evolve,
            'paroquant': cls.paroquant,
            'polymarket': cls.polymarket,
            'device': detect_device(),
        }


# ─── Status Formatter ────────────────────────────────────────────────────────

def format_status(data: dict) -> str:
    """
    Shared formatter for /status API response and Telegram /status command.
    Input:  dict from GodLocalAgent.status()
    Output: markdown-style string (works in Telegram + plain text)
    """
    lines = [
        f"GodLocal {data.get('version', 'v5')}",
        f"  Soul loaded : {data.get('soul_loaded', False)}",
        f"  Memory      : {data.get('memory_entries', 0)} entries",
        f"  LLM engine  : {data.get('llm_engine', 'unknown')}",
        f"  Device      : {data.get('device', detect_device())}",
        f"  Self-Evolve : {data.get('self_evolve', False)}",
        f"  Uptime      : {data.get('uptime', 'unknown')}",
    ]
    caps = data.get('capabilities', {})
    if caps:
        active = [k for k, v in caps.items() if v and k != 'device']
        lines.append(f"  Active caps : {', '.join(active) or 'none'}")
    return "\n".join(lines)


# ─── Atomic File Write ────────────────────────────────────────────────────────

def atomic_write(path: str, content: str) -> None:
    """
    Write content to a file atomically using tempfile + os.replace.
    Prevents data corruption on power loss or crash mid-write.
    """
    import tempfile
    dir_path = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_path, exist_ok=True)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(content)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.exception(f"atomic_write failed for {path}")
        raise
