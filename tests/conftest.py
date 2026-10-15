import io
import json
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from finrag.evaluation.import_assets import ORIGINAL_WORDING, source_paths
from finrag.ingestion.hashing import hash_bytes
from finrag.ingestion.manifest import json_bytes


def pdf_bytes(texts: list[str]) -> bytes:
    output = io.BytesIO()
    document = canvas.Canvas(output, invariant=1)
    for text in texts:
        if text:
            document.drawString(40, 700, text)
        document.showPage()
    document.save()
    return output.getvalue()


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic_report.pdf"
    path.write_bytes(pdf_bytes(["Synthetic first page", "", "Synthetic third page"]))
    return path


@pytest.fixture
def snapshot(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "prototype/evaluation"
    review = root / "external_review"
    review.mkdir(parents=True)
    documents, questions, mappings = [], [], []
    bundle_page = 0
    excerpt_extras = 13
    for code, page_count, absent_index in (("bnp", 7, 15), ("casa", 6, 10), ("sg", 6, 10)):
        digest = hash_bytes(f"synthetic-{code}".encode())
        identity = f"sha256:{digest}"
        company = f"Synthetic company {code}"
        filename = f"synthetic_{code}.pdf"
        documents.append(
            {
                "document_id": identity,
                "sha256": digest,
                "local_filename": filename,
                "company": company,
                "page_count": 100,
            }
        )
        local_mappings = []
        for page in range(1, page_count + 1):
            bundle_page += 1
            mapping = {
                "bundle_page_number": bundle_page,
                "original_physical_page": page,
                "company": company,
                "document_id": identity,
                "source_filename": filename,
                "source_sha256": digest,
                "question_ids": [],
            }
            local_mappings.append(mapping)
            mappings.append(mapping)
        for index in range(1, 16):
            identifier = f"finrag_v1_{code}_{index:02}"
            answerable = index != absent_index
            mapping = local_mappings[(index - 1) % page_count]
            physical = mapping["original_physical_page"]
            bundle = mapping["bundle_page_number"]
            evidence = []
            if answerable:
                mapping["question_ids"].append(identifier)
                evidence.append({"page_number": physical, "text": f"Synthetic evidence {bundle}."})
                if excerpt_extras:
                    evidence.append(
                        {"page_number": physical, "text": f"Synthetic secondary {bundle}."}
                    )
                    excerpt_extras -= 1
            question = {
                "question_id": identifier,
                "document_id": identity,
                "company": company,
                "language": "fr",
                "category": "factual" if answerable else "unanswerable",
                "difficulty": "easy",
                "question": f"Synthetic question {code} {index}?",
                "reference_answer": "Synthetic answer"
                if answerable
                else "No future result supplied",
                "relevant_pages": [physical] if answerable else [],
                "evidence_texts": evidence,
                "relevant_chunk_ids": [f"{identity}:p{physical}:c0:0123456789abcdef"]
                if answerable
                else [],
                "answerable": answerable,
                "annotation_status": "verified" if code == "sg" and index == 15 else "candidate",
                "annotator_note": "Synthetic annotation",
                "split": "dev" if index <= 10 else "test",
                "document_title": "Synthetic report",
                "source_url": "https://example.org/report",
                "source_sha256": digest,
                "calculation": None,
                "source_values": [],
                "unit": None,
                "currency": None,
                "period": {"values": [2025], "meaning": "Synthetic fiscal year"},
                "scope": {"kind": "company", "label": company},
                "context_annotation_status": "candidate",
                "context_note": "Synthetic context",
                "evidence_bundle_pages": [bundle] if answerable else [],
            }
            if not answerable:
                question.update(
                    corpus_years=[2025],
                    requested_year=2030,
                    possible_confusion="Future target is not an audited result",
                    reason_unanswerable="Synthetic corpus covers 2025",
                )
            if identifier == "finrag_v1_bnp_05":
                question["question"] = ORIGINAL_WORDING
            questions.append(question)
    (review / "annotations_for_review.json").write_bytes(json_bytes(questions))
    (review / "evidence_page_mapping.json").write_bytes(json_bytes(mappings))
    (review / "evidence_pages.pdf").write_bytes(
        pdf_bytes(
            [
                f"Synthetic evidence {number}. Synthetic secondary {number}."
                for number in range(1, 20)
            ]
        )
    )
    paths = source_paths(
        review / "annotations_for_review.json",
        review / "evidence_page_mapping.json",
        review / "evidence_pages.pdf",
    )
    corpus = {"documents": documents}
    config = {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "embedding_revision": "synthetic-revision",
        "index": "faiss.IndexFlatIP",
        "model_max_seq_length": 384,
        "chunking": {"max_tokens": 1250, "overlap_tokens": 0},
    }
    paths["corpus_manifest.json"].parent.mkdir(parents=True)
    paths["corpus_manifest.json"].write_bytes(json_bytes(corpus))
    paths["baseline.json"].parent.mkdir(parents=True)
    paths["baseline.json"].write_bytes(json_bytes(config))
    frozen = {
        "document_sha256": {row["document_id"]: row["sha256"] for row in documents},
        "corpus_manifest_sha256": hash_bytes(json_bytes(corpus)),
        "configuration_sha256": hash_bytes(json_bytes(config)),
        "configuration": config,
        "question_count": 45,
        "chunk_count": 3760,
        "split_counts": {"dev": 30, "test": 15},
        "source_annotation_status_counts": {"candidate": 44, "verified": 1},
        "question_ids_by_split": {
            split: [row["question_id"] for row in questions if row["split"] == split]
            for split in ("dev", "test")
        },
    }
    paths["finrag_v1.frozen.json"].parent.mkdir(parents=True)
    paths["finrag_v1.frozen.json"].write_bytes(json_bytes(frozen))
    metrics = {
        "configuration_sha256": hash_bytes(json_bytes(config)),
        "frozen_manifest_sha256": hash_bytes(json_bytes(frozen)),
    }
    paths["retrieval_baseline_metrics.json"].parent.mkdir(parents=True)
    paths["retrieval_baseline_metrics.json"].write_bytes(json_bytes(metrics))
    return paths


def change_json(path: Path, change) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_bytes(json_bytes(value))
