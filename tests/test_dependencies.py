"""Tests for api.dependencies.AppResources."""

from api.dependencies import AppResources


class TestAppResources:
    def test_defaults_none(self):
        res = AppResources()
        assert res.store is None
        assert res.embedding_model is None
        assert res.ollama_client is None
        assert res.reranker is None
        assert res.agent is None
        assert res._initialized is False

    def test_initialize_not_called_twice(self):
        """Verify that initialize is idempotent by checking the _initialized flag."""
        res = AppResources()
        res._initialized = True
        # If already initialized, initialize() should return immediately without error
        res.initialize()

    def test_shutdown_no_store(self):
        res = AppResources()
        # Should not raise when store is None
        res.shutdown()

    def test_shutdown_with_mock_store(self):
        from unittest.mock import MagicMock

        res = AppResources()
        mock_store = MagicMock()
        res.store = mock_store
        res.shutdown()
        mock_store.disconnect.assert_called_once()
