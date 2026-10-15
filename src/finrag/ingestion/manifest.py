import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from finrag.config import FinragError, filesystem_operation
from finrag.domain.documents import PageRecord
from finrag.domain.manifests import DocumentManifest


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def write_atomic(path: Path, content: bytes, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FinragError(f"{path.name}: output exists; use --overwrite explicitly.")
    if path.is_symlink() or path.is_dir():
        raise FinragError(f"{path.name}: output must be a regular file.")
    with filesystem_operation("create output directory", path.parent):
        path.parent.mkdir(parents=True, exist_ok=True)
    with filesystem_operation("create atomic temporary file", path):
        descriptor, name = tempfile.mkstemp(prefix=".finrag-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with (
            filesystem_operation("write atomic temporary file", temporary),
            os.fdopen(descriptor, "wb") as output,
        ):
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        with filesystem_operation(
            "replace output" if overwrite else "publish output without overwrite", path
        ):
            if overwrite:
                os.replace(temporary, path)
            else:
                # Hard-link publication refuses a destination created concurrently.
                os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    try:
        with filesystem_operation("read UTF-8 JSON", path):
            return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FinragError(f"{path.name}: invalid UTF-8 JSON.") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        # Unicode paragraph separators can be literal PDF text inside a JSON string.
        # Only LF separates records in files written by the UTF-8 JSONL serialiser.
        with filesystem_operation("read UTF-8 JSONL", path):
            lines = path.read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
    except UnicodeError as error:
        raise FinragError(f"{path.name}: invalid UTF-8 text.") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise FinragError(f"{path.name}: blank JSONL row at line {line_number}.")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise FinragError(f"{path.name}: invalid JSON at line {line_number}.") from error
        if not isinstance(value, dict):
            raise FinragError(f"{path.name}: line {line_number} must be a JSON object.")
        rows.append(value)
    return rows


def validate_page_sequence(pages: list[PageRecord]) -> None:
    if not pages:
        raise FinragError("Page extraction must contain at least one physical page.")
    first = pages[0]
    for expected, page in enumerate(pages, start=1):
        if page.physical_page_number != expected:
            raise FinragError(
                "Page records must include every physical page in order, including blanks."
            )
        if (
            page.document_id != first.document_id
            or page.source_sha256 != first.source_sha256
            or page.source_filename != first.source_filename
        ):
            raise FinragError("A page extraction must belong to one source document.")


def publish_directory(staged: Path, output: Path, overwrite: bool) -> None:
    backup = output.parent / f".finrag-{uuid4().hex}"
    replaced = False
    try:
        if output.exists():
            if not overwrite:
                raise FinragError("Output exists; overwrite refused.")
            os.rename(output, backup)
            replaced = True
        os.rename(staged, output)
    except Exception:
        if replaced and not output.exists():
            os.rename(backup, output)
        raise
    if replaced:
        shutil.rmtree(backup)


def save_page_extraction(
    manifest: DocumentManifest,
    pages: list[PageRecord],
    output: Path,
    overwrite: bool = False,
) -> None:
    validate_page_sequence(pages)
    if output.exists() and (not overwrite or not output.is_dir()):
        raise FinragError("Extraction output exists; use --overwrite for an existing directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".finrag-", dir=output.parent) as temporary:
        staged = Path(temporary) / "extraction"
        staged.mkdir()
        (staged / "document_manifest.json").write_bytes(
            json_bytes(manifest.model_dump(mode="json"))
        )
        (staged / "pages.jsonl").write_bytes(
            jsonl_bytes([page.model_dump(mode="json") for page in pages])
        )
        publish_directory(staged, output, overwrite)
