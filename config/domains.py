"""Insurance domain registry with URLs and metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InsuranceDomain:
    name: str
    name_he: str
    base_url: str
    description: str


DOMAINS: dict[str, InsuranceDomain] = {
    "car": InsuranceDomain(
        name="car",
        name_he="רכב",
        base_url="https://www.harel-group.co.il/insurance/car",
        description="Car insurance - comprehensive, third party, mandatory",
    ),
    "life": InsuranceDomain(
        name="life",
        name_he="חיים",
        base_url="https://www.harel-group.co.il/insurance/life",
        description="Life insurance policies and coverage",
    ),
    "travel": InsuranceDomain(
        name="travel",
        name_he="נסיעות לחו\"ל",
        base_url="https://www.harel-group.co.il/insurance/travel",
        description="Travel insurance for international trips",
    ),
    "health": InsuranceDomain(
        name="health",
        name_he="בריאות",
        base_url="https://www.harel-group.co.il/insurance/health",
        description="Health insurance - supplementary and private",
    ),
    "dental": InsuranceDomain(
        name="dental",
        name_he="שיניים",
        base_url="https://www.harel-group.co.il/insurance/dental",
        description="Dental insurance coverage",
    ),
    "mortgage": InsuranceDomain(
        name="mortgage",
        name_he="משכנתא",
        base_url="https://www.harel-group.co.il/insurance/mortgage",
        description="Mortgage insurance - life and property",
    ),
    "business": InsuranceDomain(
        name="business",
        name_he="עסקים",
        base_url="https://www.harel-group.co.il/insurance/business",
        description="Business insurance - liability, property, professional",
    ),
    "apartment": InsuranceDomain(
        name="apartment",
        name_he="דירה",
        base_url="https://www.harel-group.co.il/insurance/apartment",
        description="Apartment/home insurance - structure and contents",
    ),
}

DOMAIN_NAMES = list(DOMAINS.keys())
DOMAIN_NAMES_HE = {d.name_he: d.name for d in DOMAINS.values()}
