from pathlib import Path

from finrag.cli.main import main
from finrag.ingestion.manifest import jsonl_bytes, read_jsonl


def test_cli_extracts_and_chunks_without_printing_text(
    synthetic_pdf: Path, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "data/private/corpus/synthetic"
    assert main(["inspect-pdf", "--pdf", str(synthetic_pdf)]) == 0
    assert main(["extract-pages", "--pdf", str(synthetic_pdf), "--output", str(output)]) == 0
    assert [row["physical_page_number"] for row in read_jsonl(output / "pages.jsonl")] == [1, 2, 3]
    chunks = output / "chunks.jsonl"
    assert (
        main(
            [
                "create-chunks",
                "--pages",
                str(output / "pages.jsonl"),
                "--output",
                str(chunks),
                "--maximum-size",
                "8",
                "--overlap",
                "2",
            ]
        )
        == 0
    )
    assert (
        main(["create-chunks", "--pages", str(output / "pages.jsonl"), "--output", str(chunks)])
        == 2
    )
    captured = capsys.readouterr()
    assert "Synthetic first page" not in captured.out
    assert "Traceback" not in captured.err


def test_cli_input_errors_have_no_traceback(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["inspect-pdf", "--pdf", "missing.pdf"]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_jsonl_preserves_unicode_paragraph_separators(tmp_path: Path) -> None:
    rows = [{"text": "Fictional line\u2028paragraph\u2029next\u0085value"}, {"text": "second"}]
    path = tmp_path / "pages.jsonl"
    path.write_bytes(jsonl_bytes(rows))
    assert read_jsonl(path) == rows
