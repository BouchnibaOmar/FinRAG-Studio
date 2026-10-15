from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

APPROVED_EVALUATION_ASSETS = (
    "annotations_for_review.json",
    "evidence_page_mapping.json",
    "evidence_pages.pdf",
    "corpus_manifest.json",
    "finrag_v1.frozen.json",
    "baseline.json",
    "retrieval_baseline_metrics.json",
)


class FinragError(ValueError):
    """An input or operation violates a financial-data contract."""


class FileOperationError(FinragError):
    def __init__(self, operation: str, path: Path, error: OSError):
        self.operation = operation
        self.path = path.absolute()
        super().__init__(f"{operation} failed for '{self.path}': {type(error).__name__}: {error}")


@contextmanager
def filesystem_operation(operation: str, path: Path) -> Iterator[None]:
    try:
        yield
    except OSError as error:
        raise FileOperationError(operation, path, error) from error


def private_output(path: Path, project_root: Path) -> Path:
    if path.is_symlink():
        raise FinragError("Private output must not be a symbolic link.")
    resolved = path.resolve()
    project_root = project_root.resolve()
    roots = [project_root / "data/private", project_root / "artifacts/private"]
    if any(not root.resolve().is_relative_to(project_root) for root in roots):
        raise FinragError("Private roots must not resolve outside the project.")
    if not any(resolved.is_relative_to(root.resolve()) for root in roots):
        raise FinragError("Output must be inside data/private or artifacts/private.")
    if resolved in [root.resolve() for root in roots]:
        raise FinragError("Choose a dataset or file below the private root, not the root itself.")
    return resolved
