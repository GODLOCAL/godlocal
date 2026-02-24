"""tests/test_schemas.py — models/schemas.py Pydantic v2 validation"""
import pytest


class TestThinkRequest:
    def test_valid(self):
        from models.schemas import ThinkRequest
        r = ThinkRequest(task="hello")
        assert r.max_tokens == 2048
        assert r.long_memory is False

    def test_empty_task_raises(self):
        from models.schemas import ThinkRequest
        with pytest.raises(Exception):
            ThinkRequest(task="")

    def test_max_tokens_bounded(self):
        from models.schemas import ThinkRequest
        with pytest.raises(Exception):
            ThinkRequest(task="test", max_tokens=99999)


class TestEvolveRequest:
    def test_defaults(self):
        from models.schemas import EvolveRequest
        r = EvolveRequest(task="refactor main.py")
        assert r.apply is False
        assert r.max_revisions == 2

    def test_max_revisions_bounded(self):
        from models.schemas import EvolveRequest
        with pytest.raises(Exception):
            EvolveRequest(task="test", max_revisions=10)
