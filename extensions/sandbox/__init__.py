"""
extensions/sandbox — Docker isolation layer for AutoGenesis safe patching.
"""
from .safe_apply import DockerSafeApply

__all__ = ["DockerSafeApply"]
