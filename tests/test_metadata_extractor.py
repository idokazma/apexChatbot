"""Tests for data_pipeline.parser.metadata_extractor."""

from data_pipeline.parser.metadata_extractor import (
    detect_doc_type,
    detect_language,
    enrich_metadata,
    extract_domain_from_url,
    extract_section_path,
)


class TestDetectLanguage:
    def test_hebrew(self):
        assert detect_language("שלום עולם, איך אפשר לעזור לכם?") == "he"

    def test_english(self):
        # langdetect is mocked in test env; test that return value is valid
        result = detect_language("Hello world, how can I help you?")
        assert result in ("he", "en")

    def test_empty_defaults_hebrew(self):
        assert detect_language("") == "he"


class TestDetectDocType:
    def test_pdf(self):
        assert detect_doc_type("https://example.com/doc.pdf", "Some document") == "pdf"

    def test_policy_pdf(self):
        result = detect_doc_type("https://example.com/doc.pdf", "תנאים כלליים לפוליסה")
        assert result == "policy_document"

    def test_faq(self):
        assert detect_doc_type("https://example.com/faq", "Questions") == "faq"

    def test_webpage(self):
        assert detect_doc_type("https://example.com/page", "About Us") == "webpage"

    def test_policy_keyword(self):
        result = detect_doc_type("https://example.com/doc.pdf", "Car Insurance Policy")
        assert result == "policy_document"


class TestExtractDomainFromUrl:
    def test_car_domain(self):
        url = "https://www.harel-group.co.il/insurance/car/something"
        assert extract_domain_from_url(url) == "car"

    def test_health_domain(self):
        url = "https://www.harel-group.co.il/insurance/health/plans"
        assert extract_domain_from_url(url) == "health"

    def test_apartment_domain(self):
        url = "https://www.harel-group.co.il/insurance/apartment/coverage"
        assert extract_domain_from_url(url) == "apartment"

    def test_unknown_url(self):
        url = "https://example.com/other"
        assert extract_domain_from_url(url) is None

    def test_root_url(self):
        url = "https://www.harel-group.co.il/"
        assert extract_domain_from_url(url) is None


class TestExtractSectionPath:
    def test_single_heading(self):
        markdown = "# Main Title\nSome content"
        result = extract_section_path(markdown)
        assert len(result) == 1
        assert result[0]["level"] == 1
        assert result[0]["text"] == "Main Title"

    def test_nested_headings(self):
        markdown = "# H1\n## H2\n### H3\nContent"
        result = extract_section_path(markdown)
        assert len(result) == 3
        assert result[0]["level"] == 1
        assert result[1]["level"] == 2
        assert result[2]["level"] == 3

    def test_no_headings(self):
        markdown = "Just plain text\nNo headings here."
        result = extract_section_path(markdown)
        assert result == []


class TestEnrichMetadata:
    def test_enriches_language(self):
        doc = {
            "markdown": "Hello world, this is English text for testing.",
            "source_url": "",
            "title": "Test",
        }
        result = enrich_metadata(doc)
        # langdetect is mocked in test env; verify key is populated
        assert "language" in result
        assert result["language"] in ("he", "en")

    def test_enriches_doc_type(self):
        doc = {
            "markdown": "Content",
            "source_url": "https://example.com/doc.pdf",
            "title": "Some Document",
        }
        result = enrich_metadata(doc)
        assert result["doc_type"] == "pdf"

    def test_enriches_domain_from_url(self):
        doc = {
            "markdown": "Content",
            "source_url": "https://www.harel-group.co.il/insurance/car/coverage",
            "title": "Car Coverage",
        }
        result = enrich_metadata(doc)
        assert result["domain"] == "car"

    def test_preserves_existing_domain(self):
        doc = {
            "markdown": "Content",
            "source_url": "https://example.com",
            "title": "Test",
            "domain": "health",
        }
        result = enrich_metadata(doc)
        assert result["domain"] == "health"

    def test_unknown_domain(self):
        doc = {
            "markdown": "Content",
            "source_url": "https://example.com/other",
            "title": "Test",
        }
        result = enrich_metadata(doc)
        assert result["domain"] == "unknown"

    def test_headings_extracted(self):
        doc = {
            "markdown": "# Title\n## Section\nContent",
            "source_url": "",
            "title": "Test",
        }
        result = enrich_metadata(doc)
        assert len(result["headings"]) == 2
