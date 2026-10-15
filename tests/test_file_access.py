import ctypes
import errno
import os
from pathlib import Path

import pytest

from finrag.cli.main import main
from finrag.config import FileOperationError
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, read_json, write_atomic


def test_read_denial_retains_operation_path_message_and_cause(tmp_path, monkeypatch):
    path = tmp_path / "sample_benchmark.json"
    original = PermissionError(errno.EACCES, "locked by another handle", str(path))

    def denied(*args, **kwargs):
        raise original

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(FileOperationError) as caught:
        read_json(path)
    assert caught.value.__cause__ is original
    message = str(caught.value)
    assert "read UTF-8 JSON" in message and str(path) in message
    assert "PermissionError" in message and "locked by another handle" in message


def test_hash_directory_reports_read_operation_and_original_error(tmp_path):
    with pytest.raises(FileOperationError) as caught:
        hash_file(tmp_path)
    assert isinstance(caught.value.__cause__, OSError)
    assert "read file for SHA-256" in str(caught.value)
    assert str(tmp_path) in str(caught.value)


def test_atomic_publication_denial_keeps_original_and_existing_files(tmp_path, monkeypatch):
    path = tmp_path / "batch-000003.npy"
    existing = tmp_path / "batch-000000.npy"
    existing.write_bytes(b"existing checkpoint")
    original = PermissionError(errno.EACCES, "publication denied", str(path))

    def denied(*args, **kwargs):
        raise original

    monkeypatch.setattr(os, "link", denied)
    with pytest.raises(FileOperationError) as caught:
        write_atomic(path, b"synthetic test")
    assert caught.value.__cause__ is original
    assert "publish output without overwrite" in str(caught.value)
    assert str(path) in str(caught.value)
    assert existing.read_bytes() == b"existing checkpoint"
    assert not path.exists() and not list(tmp_path.glob(".finrag-*.tmp"))


def test_cli_keeps_original_os_error_details(monkeypatch, capsys):
    import importlib

    cli = importlib.import_module("finrag.cli.main")

    def denied(args):
        raise PermissionError(errno.EACCES, "sharing violation", "blocked.pdf")

    monkeypatch.setattr(cli, "run_command", denied)
    assert main(["inspect-pdf", "--pdf", "blocked.pdf"]) == 2
    message = capsys.readouterr().err
    assert all(
        item in message
        for item in ("denied", "blocked.pdf", "PermissionError", "sharing violation")
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows exclusive-handle regression")
def test_startup_check_does_not_touch_exclusively_locked_manual_log(tmp_path, monkeypatch):
    from test_dense_retrieval import model_chunks

    from finrag.evaluation import run_dense_comparison as runner
    from finrag.retrieval.embedding_cache import EmbeddingCache

    root = tmp_path / "artifacts/private/phase2b"
    root.mkdir(parents=True)
    chunks = model_chunks(["fictional revenue"])
    write_atomic(
        root / "sample_benchmark.json",
        json_bytes({"fingerprint": {}, "requires_manual_execution": True}),
    )
    EmbeddingCache(root / "embeddings", {}, len(chunks))
    log = root / "manual_run.log"
    log.write_bytes(b"existing redirected output")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "E5Encoder", lambda *args: object())
    monkeypatch.setattr(runner, "prepare_dense_inputs", lambda *args: (chunks, {}))
    monkeypatch.setattr(runner, "encoding_fingerprint", lambda *args: {})

    def forbidden(*args, **kwargs):
        pytest.fail("Encoding must not run during the startup file check")

    monkeypatch.setattr(EmbeddingCache, "resume", forbidden)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    from ctypes import wintypes

    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(log), 0x40000000, 0, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value
    try:
        with pytest.raises(PermissionError):
            log.read_bytes()
        result = runner.run_comparison(tmp_path, root, "cpu", 4, check_files=True)
        assert result["startup_file_validation"] == "passed"
        assert result["encoding_started"] is False
    finally:
        kernel.CloseHandle(handle)
    assert log.read_bytes() == b"existing redirected output"
