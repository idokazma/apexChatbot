"""Tests for agent.nodes.retriever_node."""

from unittest.mock import MagicMock

from agent.nodes.retriever_node import retriever_node


class TestRetrieverNode:
    def test_uses_rewritten_query(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [{"content": "result"}]
        state = {
            "query": "original",
            "rewritten_query": "rewritten query",
            "detected_domains": ["car"],
        }
        result = retriever_node(state, retriever)
        retriever.retrieve.assert_called_once_with(query="rewritten query", domains=["car"])
        assert result["retrieved_documents"] == [{"content": "result"}]

    def test_falls_back_to_original_query(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        state = {"query": "original query", "rewritten_query": "", "detected_domains": []}
        result = retriever_node(state, retriever)
        retriever.retrieve.assert_called_once_with(query="original query", domains=None)
        assert result["retrieved_documents"] == []

    def test_no_domains_passes_none(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        state = {"query": "test", "rewritten_query": "", "detected_domains": []}
        retriever_node(state, retriever)
        retriever.retrieve.assert_called_with(query="test", domains=None)

    def test_multiple_domains_passed(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        state = {
            "query": "test",
            "rewritten_query": "",
            "detected_domains": ["car", "health"],
        }
        retriever_node(state, retriever)
        retriever.retrieve.assert_called_with(query="test", domains=["car", "health"])
