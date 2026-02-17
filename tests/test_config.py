"""Tests for config.settings and config.domains."""

from config.domains import DOMAIN_NAMES, DOMAIN_NAMES_HE, DOMAINS, InsuranceDomain
from config.settings import Settings


class TestInsuranceDomain:
    def test_domain_is_frozen(self):
        domain = InsuranceDomain(
            name="test", name_he="טסט", base_url="https://example.com", description="Test"
        )
        assert domain.name == "test"
        # Frozen dataclass should raise on mutation
        try:
            domain.name = "changed"
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass

    def test_domain_fields(self):
        domain = DOMAINS["car"]
        assert domain.name == "car"
        assert domain.name_he == "רכב"
        assert "harel-group" in domain.base_url
        assert domain.description != ""


class TestDomainRegistry:
    def test_all_eight_domains_exist(self):
        expected = {"car", "life", "travel", "health", "dental", "mortgage", "business", "apartment"}
        assert set(DOMAINS.keys()) == expected

    def test_domain_names_list(self):
        assert len(DOMAIN_NAMES) == 8
        assert "car" in DOMAIN_NAMES
        assert "apartment" in DOMAIN_NAMES

    def test_domain_names_he_mapping(self):
        assert DOMAIN_NAMES_HE["רכב"] == "car"
        assert DOMAIN_NAMES_HE["בריאות"] == "health"
        assert DOMAIN_NAMES_HE["דירה"] == "apartment"
        assert len(DOMAIN_NAMES_HE) == 8

    def test_all_domains_have_base_url(self):
        for name, domain in DOMAINS.items():
            assert domain.base_url.startswith("https://"), f"{name} missing HTTPS base_url"

    def test_all_domains_have_hebrew_name(self):
        for name, domain in DOMAINS.items():
            assert domain.name_he, f"{name} missing Hebrew name"


class TestSettings:
    def test_default_values(self):
        s = Settings(anthropic_api_key="", openai_api_key="")
        assert s.ollama_host == "http://localhost:11434"
        assert s.ollama_model == "gemma3:12b"
        assert s.embedding_dim == 1024
        assert s.top_k_retrieve == 10
        assert s.top_k_rerank == 5
        assert s.max_chunk_tokens == 512
        assert s.chunk_overlap_tokens == 50
        assert s.log_level == "INFO"

    def test_settings_types(self):
        s = Settings(anthropic_api_key="", openai_api_key="")
        assert isinstance(s.top_k_retrieve, int)
        assert isinstance(s.embedding_dim, int)
        assert isinstance(s.ollama_host, str)
