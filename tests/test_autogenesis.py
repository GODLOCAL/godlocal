"""tests/test_autogenesis.py — agents/autogenesis_v2.py coverage"""
import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class TestFEPState:
    def test_initial_state(self):
        from agents.autogenesis_v2 import FEPState
        fep = FEPState()
        assert fep.free_energy == 1.0
        assert fep.correction_rate == 0.0
        assert fep.total_interactions == 0

    def test_update_corrected(self):
        from agents.autogenesis_v2 import FEPState
        fep = FEPState()
        fep.update(corrected=True)
        fep.update(corrected=True)
        fep.update(corrected=False)
        assert fep.total_interactions == 3
        assert abs(fep.correction_rate - 2/3) < 0.01
        assert abs(fep.free_energy - (1 - 2/3)) < 0.01

    def test_file_changed_detection(self):
        from agents.autogenesis_v2 import FEPState
        fep = FEPState()
        assert fep.file_changed("test.py", "hash1") is True   # first time: changed
        assert fep.file_changed("test.py", "hash1") is False  # same hash
        assert fep.file_changed("test.py", "hash2") is True   # different hash

    def test_to_dict_keys(self):
        from agents.autogenesis_v2 import FEPState
        fep = FEPState()
        d = fep.to_dict()
        assert "correction_rate" in d
        assert "free_energy" in d
        assert "evolutions" in d


class TestCodeScanner:
    def test_scan_finds_py_files(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hi')")
        (tmp_path / "README.md").write_text("# test")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"")

        from agents.autogenesis_v2 import CodeScanner
        scanner = CodeScanner()
        files = scanner.scan(str(tmp_path))
        paths = [f.name for f in files]
        assert "main.py" in paths
        assert "README.md" in paths
        # pycache excluded
        assert not any("cpython" in p for p in paths)

    def test_hash_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("print('hello')")
        from agents.autogenesis_v2 import CodeScanner
        scanner = CodeScanner()
        h = scanner.hash_file(f)
        assert isinstance(h, str)
        assert len(h) == 12


class TestSearchReplacePatcher:
    def test_applies_single_patch(self):
        from agents.autogenesis_v2 import SearchReplacePatcher
        p = SearchReplacePatcher()
        content = "def foo():
    return 1
"
        patch_text = "<<<SEARCH
    return 1
>>>
<<<REPLACE
    return 42
>>>"
        result, count = p.apply(content, patch_text)
        assert count == 1
        assert "return 42" in result

    def test_applies_multiple_patches(self):
        from agents.autogenesis_v2 import SearchReplacePatcher
        p = SearchReplacePatcher()
        content = "a = 1
b = 2
"
        patch_text = (
            "<<<SEARCH
a = 1
>>>
<<<REPLACE
a = 10
>>>
"
            "<<<SEARCH
b = 2
>>>
<<<REPLACE
b = 20
>>>"
        )
        result, count = p.apply(content, patch_text)
        assert count == 2
        assert "a = 10" in result
        assert "b = 20" in result

    def test_missing_search_skips(self):
        from agents.autogenesis_v2 import SearchReplacePatcher
        p = SearchReplacePatcher()
        content = "original content"
        patch_text = "<<<SEARCH
NOT IN FILE
>>>
<<<REPLACE
replacement
>>>"
        result, count = p.apply(content, patch_text)
        assert count == 0
        assert result == content

    def test_no_patches_returns_original(self):
        from agents.autogenesis_v2 import SearchReplacePatcher
        p = SearchReplacePatcher()
        content = "unchanged"
        result, count = p.apply(content, "no patches here")
        assert count == 0
        assert result == content


class TestAutoGenesis:
    def test_init(self, tmp_root):
        from agents.autogenesis_v2 import AutoGenesis
        ag = AutoGenesis(root=str(tmp_root))
        assert ag.fep is not None
        assert ag.scanner is not None
        assert ag.patcher is not None

    def test_record_correction(self, tmp_root):
        from agents.autogenesis_v2 import AutoGenesis
        ag = AutoGenesis(root=str(tmp_root))
        ag.record_correction(True)
        ag.record_correction(False)
        metrics = ag.fep_metrics()
        assert metrics["total_interactions"] == 2
        assert metrics["correction_rate"] == 0.5

    def test_fep_metrics_returns_dict(self, tmp_root):
        from agents.autogenesis_v2 import AutoGenesis
        ag = AutoGenesis(root=str(tmp_root))
        m = ag.fep_metrics()
        assert isinstance(m, dict)
        assert "free_energy" in m

    @pytest.mark.asyncio
    async def test_evolve_async_dry_run(self, tmp_root):
        """Dry run should return result without modifying files."""
        from agents.autogenesis_v2 import AutoGenesis
        from core.brain import Brain

        mock_brain = MagicMock()
        mock_brain.think = AsyncMock(return_value=json.dumps({
            "mode": "FEATURE",
            "target_files": ["test_target.py"],
            "changes": ["add docstring"],
            "risk": "low",
            "test_command": "pytest tests/"
        }))
        mock_brain.memory = MagicMock()
        mock_brain.memory.query = MagicMock(return_value=[])
        mock_brain.memory.add = MagicMock()
        mock_brain.memory.prune = MagicMock(return_value=0)

        (tmp_root / "test_target.py").write_text("def foo(): pass
")

        with patch.object(Brain, "get", return_value=mock_brain):
            ag = AutoGenesis(root=str(tmp_root))
            result = await ag.evolve_async(task="Add docstring to foo", apply=False)
            assert result["status"] in ("dry_run", "no_targets", "evolved")

    @pytest.mark.asyncio
    async def test_evolve_high_risk_skipped_without_apply(self, tmp_root):
        from agents.autogenesis_v2 import AutoGenesis
        from core.brain import Brain

        mock_brain = MagicMock()
        mock_brain.think = AsyncMock(return_value=json.dumps({
            "mode": "REFACTOR",
            "target_files": [],
            "changes": ["massive refactor"],
            "risk": "high",
        }))
        mock_brain.memory = MagicMock()
        mock_brain.memory.query = MagicMock(return_value=[])
        mock_brain.memory.add = MagicMock()
        mock_brain.memory.prune = MagicMock(return_value=0)

        with patch.object(Brain, "get", return_value=mock_brain):
            ag = AutoGenesis(root=str(tmp_root))
            result = await ag.evolve_async(task="risky task", apply=False)
            assert result["status"] == "skipped"
            assert "high-risk" in result["reason"]
