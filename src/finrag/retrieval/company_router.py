import re
import unicodedata
from typing import Literal

from finrag.config import FinragError
from finrag.domain.documents import NonEmptyText, SourceIdentity, StrictRecord

Company = Literal["BNP Paribas", "Crédit Agricole", "Société Générale"]


class CatalogueDocument(SourceIdentity):
    company_family: Company
    reporting_entity: NonEmptyText
    language: Literal["fr", "en"]


class CompanyRoute(StrictRecord):
    companies: list[Company]
    document_ids: list[str]
    triggering_terms: dict[str, list[str]]
    status: Literal["single_company", "multiple_companies", "no_company"]
    scope_notice: str = "Document selection does not verify the financial entity scope."


def normalize_company_text(text: str) -> str:
    folded = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", text).casefold())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", folded))


def route_company(question: str, catalogue: list[CatalogueDocument]) -> CompanyRoute:
    if not catalogue or len({doc.document_id for doc in catalogue}) != len(catalogue):
        raise FinragError("Company routing requires a nonempty catalogue with unique documents.")
    normalized = " " + normalize_company_text(question) + " "
    aliases = {
        "BNP Paribas": ("bnp paribas", "b n p paribas"),
        "Crédit Agricole": ("credit agricole",),
        "Société Générale": ("societe generale",),
    }
    terms = {
        company: [alias for alias in names if " " + alias + " " in normalized]
        for company, names in aliases.items()
    }
    terms = {company: matched for company, matched in terms.items() if matched}
    companies = list(terms)
    documents = sorted(
        doc.document_id for doc in catalogue if not companies or doc.company_family in companies
    )
    return CompanyRoute(
        companies=companies,
        document_ids=documents,
        triggering_terms=terms,
        status="no_company"
        if not companies
        else "single_company"
        if len(companies) == 1
        else "multiple_companies",
    )
