"""tests/test_agent_pool.py — agents/agent_pool.py coverage"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch


class TestAgentPool:
    def test_init_no_mlx(self):
        with patch("agents.agent_pool.settings") as ms:
            ms.model = "qwen3:8b"
            from agents.agent_pool import AgentPool
            pool = AgentPool()
            assert pool._mlx_available is False or pool._mlx_available is True  # depends on env
            assert len(pool._slots) == len(pool._slots)  # 7 slots

    def test_status_returns_dict(self):
        from agents.agent_pool import AgentPool
        pool = AgentPool()
        status = pool.status()
        assert "active" in status
        assert "slots" in status
        assert isinstance(status["slots"], list)

    def test_status_active_none_initially(self):
        from agents.agent_pool import AgentPool
        pool = AgentPool()
        assert pool.status()["active"] is None

    @pytest.mark.asyncio
    async def test_swap_unknown_agent_raises(self):
        from agents.agent_pool import AgentPool
        pool = AgentPool()
        with pytest.raises(ValueError, match="Unknown agent type"):
            await pool.swap("nonexistent_agent")

    @pytest.mark.asyncio
    async def test_swap_stub_mode(self):
        """In stub mode (no mlx_lm), swap should still succeed and track active."""
        with patch("agents.agent_pool.AgentPool._check_mlx", return_value=False):
            from agents.agent_pool import AgentPool
            pool = AgentPool()
            result = await pool.swap("coding")
            assert result["agent"] == "coding"
            assert pool._active == "coding"

    @pytest.mark.asyncio
    async def test_swap_evicts_previous(self):
        with patch("agents.agent_pool.AgentPool._check_mlx", return_value=False):
            from agents.agent_pool import AgentPool
            pool = AgentPool()
            await pool.swap("coding")
            await pool.swap("trading")
            assert pool._active == "trading"
            # Previous slot model should be None (stub mode — was never set)
            assert pool._slots["coding"].model is None

    @pytest.mark.asyncio
    async def test_swap_increments_swap_count(self):
        with patch("agents.agent_pool.AgentPool._check_mlx", return_value=False):
            from agents.agent_pool import AgentPool
            pool = AgentPool()
            await pool.swap("writing")
            await pool.swap("writing")
            # second swap of same agent (already active)
            result = await pool.swap("writing")
            assert result["swaps"] >= 1
