"""Tests for API request/response schemas."""

import pytest
from pydantic import ValidationError

from api.schemas import ChatRequest, ChatResponse, Citation, HealthResponse


class TestChatRequest:
    def test_valid_request(self):
        req = ChatRequest(message="What is car insurance?")
        assert req.message == "What is car insurance?"
        assert req.conversation_id is None
        assert req.language is None

    def test_with_all_fields(self):
        req = ChatRequest(message="Hello", conversation_id="abc123", language="he")
        assert req.conversation_id == "abc123"
        assert req.language == "he"

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_message_max_length(self):
        long_msg = "a" * 2001
        with pytest.raises(ValidationError):
            ChatRequest(message=long_msg)

    def test_message_at_max_length(self):
        msg = "a" * 2000
        req = ChatRequest(message=msg)
        assert len(req.message) == 2000


class TestCitation:
    def test_defaults(self):
        cit = Citation()
        assert cit.source_url == ""
        assert cit.document_title == ""
        assert cit.section == ""
        assert cit.relevant_text == ""
        assert cit.page_number == 0
        assert cit.source_file_path == ""

    def test_with_values(self):
        cit = Citation(
            source_url="https://example.com",
            document_title="Test Doc",
            section="Section 1",
            relevant_text="Some text",
            page_number=5,
            source_file_path="car/files/doc.pdf",
        )
        assert cit.source_url == "https://example.com"
        assert cit.page_number == 5


class TestChatResponse:
    def test_minimal_response(self):
        resp = ChatResponse(answer="Hello")
        assert resp.answer == "Hello"
        assert resp.citations == []
        assert resp.domain is None
        assert resp.confidence == 0.0
        assert resp.conversation_id == ""
        assert resp.language == "he"

    def test_full_response(self):
        cit = Citation(source_url="https://example.com", document_title="Doc")
        resp = ChatResponse(
            answer="The coverage includes...",
            citations=[cit],
            domain="car",
            confidence=0.9,
            conversation_id="conv123",
            language="en",
        )
        assert len(resp.citations) == 1
        assert resp.domain == "car"
        assert resp.confidence == 0.9


class TestHealthResponse:
    def test_healthy(self):
        resp = HealthResponse(
            status="healthy",
            vector_db=True,
            ollama=True,
            embedding_model=True,
            collection_count=1000,
        )
        assert resp.status == "healthy"
        assert resp.collection_count == 1000

    def test_degraded(self):
        resp = HealthResponse(
            status="degraded",
            vector_db=True,
            ollama=False,
            embedding_model=True,
        )
        assert resp.status == "degraded"
        assert resp.collection_count == 0
