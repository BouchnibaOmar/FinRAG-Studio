import io
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from finrag.config import FinragError, filesystem_operation
from finrag.domain.dense import ModelChunk
from finrag.domain.retrieval import LexicalConfig
from finrag.evaluation.question_translation import TranslationInput
from finrag.evaluation.translation_cache import load_translation_set
from finrag.ingestion.hashing import hash_bytes, hash_file
from finrag.ingestion.manifest import read_json, read_jsonl
from finrag.retrieval.candidate_reranking import RerankInput, input_identity
from finrag.retrieval.company_router import CatalogueDocument
from finrag.retrieval.local_reranker import validate_local_record
from finrag.retrieval.routed_bm25 import CachedQueryTranslation, RoutedBM25, RoutedRanking

CATALOGUE = "experiments/company_routing/corpus_catalog.json"
TRANSLATION_CACHE = "artifacts/private/bm25_query_translation_dev/translator/cache"
TRANSLATION_SET = "sets/00bf6cd63861496ea3bae30f1858e7d7.json"
PREPARED = "artifacts/private/bm25_candidate_reranking_dev"
ROUTING = "artifacts/private/bm25_company_routing_dev/run_1/predictions_top50.jsonl"


def load_catalogue(project: Path) -> list[CatalogueDocument]:
    catalogue = [CatalogueDocument.model_validate(row) for row in read_json(project / CATALOGUE)]
    if not catalogue or len({doc.document_id for doc in catalogue}) != len(catalogue):
        raise FinragError("Le catalogue doit contenir des documents uniques.")
    return catalogue


def load_translations(project: Path) -> list[CachedQueryTranslation]:
    root = project / TRANSLATION_CACHE
    records = load_translation_set(root, root / TRANSLATION_SET)
    return [
        CachedQueryTranslation(
            original_question=TranslationInput.model_validate(r["input"]).question,
            translated_question=r["translation"],
            target_language=r["input"]["target_language"],
            provenance=r["source"],
        )
        for r in records.values()
    ]


class FreeSearch:
    def __init__(self, search: RoutedBM25, notices: list[str]):
        self.search_engine = search
        self.notices = notices
        # Streamlit shares cached indexes across sessions; index creation is mutable.
        self.lock = RLock()

    def search(self, question: str) -> RoutedRanking:
        if not question.strip():
            raise FinragError("Saisissez une question avant de rechercher.")
        # Preserve exact whitespace for cache matching; strip only for the empty check.
        with self.lock:
            return self.search_engine.search(question, use_cached_translation=True)


def load_free_search(project: Path) -> FreeSearch:
    catalogue = load_catalogue(project)
    chunks_path = project / "artifacts/private/phase2b/model_chunks.jsonl"
    request = read_json(project / "artifacts/private/phase2b/faiss_exchange/request.json")
    if hash_file(chunks_path) != request["inputs"]["chunks"]["sha256"]:
        raise FinragError("L’empreinte du corpus de passages diffère du cache scellé.")
    chunks = [ModelChunk.model_validate(row) for row in read_jsonl(chunks_path)]
    notices = []
    try:
        translations = load_translations(project)
    except (FinragError, OSError) as error:
        translations = []
        notices.append(
            "Traductions en cache indisponibles : recherche avec la question originale. "
            + str(error)
        )
    config = LexicalConfig.model_validate(read_json(project / "experiments/phase2a/configs/B.json"))
    return FreeSearch(RoutedBM25(chunks, catalogue, config, translations), notices)


@dataclass(frozen=True)
class SavedExample:
    question_id: str
    original_question: str
    query_used: str
    companies: list[str]
    bm25: list[dict]
    bge: list[dict]


def assemble_example(
    baseline: dict,
    route: dict,
    source: RerankInput,
    record: dict,
    translations: list[CachedQueryTranslation],
) -> SavedExample:
    if record["status"] != "success":
        raise FinragError("Le classement BGE sauvegardé n’est pas un succès.")
    if baseline["input_id"] != input_identity(source) or route["query_used"] != source.question:
        raise FinragError("La requête sauvegardée ne correspond pas aux candidats scellés.")
    hits = baseline["hits"]
    if hits != route["hits"] or len(hits) != 50:
        raise FinragError("Les candidats BM25 sauvegardés diffèrent.")
    by_id = {h["chunk_id"]: h for h in hits}
    mapping = baseline["candidate_id_to_chunk_id"]
    if len(by_id) != 50 or set(mapping) != {c.candidate_id for c in source.candidates}:
        raise FinragError("Les identifiants des candidats sauvegardés diffèrent.")
    for candidate in source.candidates:
        hit = by_id[mapping[candidate.candidate_id]]
        if (
            hit["text"] != candidate.text
            or hit["physical_page_number"] != candidate.physical_page_number
            or any(
                hit[k] != getattr(candidate.document, k)
                for k in ("document_id", "source_filename", "source_sha256")
            )
        ):
            raise FinragError("La provenance d’un candidat sauvegardé diffère.")
    ranking = validate_local_record(source, record, record["configuration"])
    original = route["original_question"]
    if original != source.question and not any(
        t.original_question == original
        and t.translated_question == source.question
        and t.target_language in {c.document.language for c in source.candidates}
        for t in translations
    ):
        raise FinragError("La traduction sauvegardée ne correspond pas à la question originale.")
    return SavedExample(
        baseline["question_id"],
        original,
        source.question,
        route["route"]["companies"],
        [dict(h) for h in hits],
        [
            {**by_id[mapping[r["candidate_id"]]], "rank": r["rank"], "bge_score": r["score"]}
            for r in ranking
        ],
    )


def load_examples(project: Path) -> list[SavedExample]:
    catalogue = {doc.document_id: doc for doc in load_catalogue(project)}
    prepared = project / PREPARED
    manifest = read_json(prepared / "manifest.json")
    if manifest["split"] != "dev" or manifest["variant"] != "cached_translation":
        raise FinragError("Seuls les exemples dev cached_translation sont autorisés.")
    for name in ("reranker/inputs.jsonl", "evaluation/baseline.jsonl"):
        expected = manifest["prepared_hashes"].get(str(Path(name)))
        if hash_file(prepared / name) != expected:
            raise FinragError("Les entrées précalculées ont changé d’empreinte.")
    route_path = project / ROUTING
    expected = manifest["source_hashes"].get(str(route_path))
    if hash_file(route_path) != expected:
        raise FinragError("Les prédictions de routage ont changé d’empreinte.")
    sources = {
        input_identity(s): s
        for s in (
            RerankInput.model_validate(row)
            for row in read_jsonl(prepared / "reranker/inputs.jsonl")
        )
    }
    if any(
        catalogue.get(candidate.document.document_id) != candidate.document
        for source in sources.values()
        for candidate in source.candidates
    ):
        raise FinragError("Les documents des exemples diffèrent du catalogue autorisé.")
    # These projections use saved results only. Metrics and annotations are never consulted.
    baselines = read_jsonl(prepared / "evaluation/baseline.jsonl")
    routes = {r["question_id"]: r for r in read_jsonl(route_path)}
    translations = load_translations(project)
    originals = {t.translated_question: t.original_question for t in translations}
    records = {}
    for path in (project / "artifacts/private/bge_reranker_dev/cache_cpu").glob("*.json"):
        record = read_json(path)
        if record["input_id"] in records:
            raise FinragError("Plusieurs caches BGE correspondent à la même entrée.")
        records[record["input_id"]] = record
    if len(baselines) != 28 or len(sources) != 28 or len(routes) != 28:
        raise FinragError("Les exemples doivent contenir les 28 questions dev sauvegardées.")
    examples = []
    for baseline in baselines:
        source = sources[baseline["input_id"]]
        row = routes[baseline["question_id"]]
        original_query = row["original"]["query_used"]
        cached = row["cached_translation"]
        if original_query != originals.get(source.question, original_query):
            raise FinragError("La question originale et sa traduction diffèrent des caches.")
        examples.append(
            assemble_example(
                baseline,
                {**cached, "original_question": original_query},
                source,
                records[baseline["input_id"]],
                translations,
            )
        )
    return sorted(examples, key=lambda e: e.question_id)


def authorized_pdf_path(
    project: Path,
    catalogue: list[CatalogueDocument],
    document_id: str,
) -> Path:
    matching = [doc for doc in catalogue if doc.document_id == document_id]
    if len(matching) != 1:
        raise FinragError("Ce document ne figure pas dans le catalogue autorisé.")
    filename = matching[0].source_filename
    if any(c in filename for c in ("/", "\\", ":", "\x00")) or not filename.endswith(".pdf"):
        raise FinragError("Le catalogue contient un nom de PDF non autorisé.")
    root = (project / "data/private/corpus/originals").resolve()
    path = (root / filename).resolve()
    if not root.is_relative_to(project.resolve()) or not path.is_relative_to(root):
        raise FinragError("Le PDF doit rester dans le répertoire local autorisé.")
    return path


def read_source_pdf(
    project: Path,
    catalogue: list[CatalogueDocument],
    document_id: str,
    page: int,
) -> tuple[bytes, str]:
    if type(page) is not int or page < 1:
        raise FinragError("La page physique doit être un entier positif.")
    path = authorized_pdf_path(project, catalogue, document_id)
    with filesystem_operation("lire le PDF local autorisé", path):
        data = path.read_bytes()
    doc = next(doc for doc in catalogue if doc.document_id == document_id)
    if hash_bytes(data) != doc.source_sha256:
        raise FinragError("Le PDF local ne correspond pas à l’empreinte du document source.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if page > len(reader.pages):
            raise FinragError("La page physique demandée dépasse le nombre de pages du PDF.")
    except PdfReadError as error:
        raise FinragError("Le PDF local est illisible ou chiffré.") from error
    return data, doc.source_filename
