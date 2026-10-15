import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from finrag.config import FinragError, private_output
from finrag.domain.documents import NonEmptyText, PositiveInt, SourceIdentity, StrictRecord
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, publish_directory


class ReportSpec(SourceIdentity):
    company: NonEmptyText
    historical_physical_pages: PositiveInt
    historical_text_pages: PositiveInt


REPORTS = [
    ReportSpec(
        document_id="sha256:65822993e7ad299d84f16bd4b495391bb670800bb9a0f301ef72e97b6f62a2e2",
        source_sha256="65822993e7ad299d84f16bd4b495391bb670800bb9a0f301ef72e97b6f62a2e2",
        source_filename="bnp_urd_2025_en.pdf",
        company="BNP Paribas",
        historical_physical_pages=936,
        historical_text_pages=934,
    ),
    ReportSpec(
        document_id="sha256:d5aa05cbf9c4840d0c475cd6321abf0134f9d444f6e01bc887ad7d0238b89fa2",
        source_sha256="d5aa05cbf9c4840d0c475cd6321abf0134f9d444f6e01bc887ad7d0238b89fa2",
        source_filename="casa_urd_2025_en.pdf",
        company="Crédit Agricole S.A.",
        historical_physical_pages=1082,
        historical_text_pages=1081,
    ),
    ReportSpec(
        document_id="sha256:bb6c44ca05dd32593a493a8b0deb4f8e82e2f7ed33ac611e9552a3a58641fd54",
        source_sha256="bb6c44ca05dd32593a493a8b0deb4f8e82e2f7ed33ac611e9552a3a58641fd54",
        source_filename="sg_urd_2026_fy2025_fr.pdf",
        company="Société Générale",
        historical_physical_pages=716,
        historical_text_pages=714,
    ),
]


class ImportedReport(ReportSpec):
    source_path: str
    destination_path: str
    byte_size: PositiveInt


class CorpusManifest(StrictRecord):
    schema_version: str = "1.0"
    creation_timestamp: datetime
    reports: list[ImportedReport]


def import_corpus(
    source_directory: Path,
    output: Path,
    project_root: Path,
    overwrite: bool = False,
    reports: list[ReportSpec] | None = None,
) -> CorpusManifest:
    output = private_output(output, project_root)
    reports = REPORTS if reports is None else reports
    if not reports or len({report.document_id for report in reports}) != len(reports):
        raise FinragError("Corpus specifications require unique document IDs.")
    if len({report.source_filename for report in reports}) != len(reports):
        raise FinragError("Corpus specifications require unique source filenames.")
    if output.exists() and (not overwrite or not output.is_dir()):
        raise FinragError("Corpus destination exists; overwrite is refused by default.")
    sources = [source_directory / report.source_filename for report in reports]
    if any(path.resolve().is_relative_to(output) for path in sources):
        raise FinragError("Corpus destination must not contain an original source.")
    for source, report in zip(sources, reports, strict=True):
        if hash_file(source) != report.source_sha256:
            raise FinragError(f"{source.name}: corpus source hash mismatch.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finrag-", dir=output.parent) as temporary:
        staged = Path(temporary) / "corpus"
        staged.mkdir()
        records = []
        for source, report in zip(sources, reports, strict=True):
            destination = staged / report.source_filename
            shutil.copyfile(source, destination)
            if (
                hash_file(destination) != report.source_sha256
                or hash_file(source) != report.source_sha256
            ):
                raise FinragError(f"{source.name}: corpus bytes changed during copy.")
            records.append(
                ImportedReport(
                    **report.model_dump(),
                    source_path=str(source.resolve()),
                    destination_path=str(output / report.source_filename),
                    byte_size=destination.stat().st_size,
                )
            )
        manifest = CorpusManifest(creation_timestamp=datetime.now(UTC), reports=records)
        (staged / "corpus_manifest.json").write_bytes(json_bytes(manifest.model_dump(mode="json")))
        publish_directory(staged, output, overwrite)
    validate_corpus(output / "corpus_manifest.json")
    return manifest


def validate_corpus(path: Path) -> CorpusManifest:
    manifest = CorpusManifest.model_validate_json(path.read_bytes())
    ids, names = set(), set()
    for report in manifest.reports:
        if report.document_id in ids or report.source_filename in names:
            raise FinragError("Corpus manifest contains duplicate documents or filenames.")
        ids.add(report.document_id)
        names.add(report.source_filename)
        destination = path.parent / report.source_filename
        if destination.resolve() != Path(report.destination_path).resolve():
            raise FinragError("Corpus manifest destination does not match its private directory.")
        if (
            hash_file(destination) != report.source_sha256
            or destination.stat().st_size != report.byte_size
        ):
            raise FinragError(f"{report.source_filename}: imported corpus hash/size mismatch.")
    if not manifest.reports:
        raise FinragError("Corpus manifest is empty.")
    return manifest
