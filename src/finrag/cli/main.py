import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from finrag.chunking.page_chunker import chunk_pages
from finrag.config import FinragError, private_output
from finrag.corpus.import_corpus import import_corpus
from finrag.corpus.prepare_corpus import prepare_chunks, prepare_extraction
from finrag.domain.chunks import ChunkingConfig
from finrag.domain.documents import PageRecord
from finrag.evaluation.import_assets import import_evaluation, source_paths
from finrag.evaluation.run_lexical_experiment import load_config, publish_dev_report, run_experiment
from finrag.evaluation.validate_dataset import validate_dataset, validate_record
from finrag.ingestion.manifest import jsonl_bytes, read_jsonl, save_page_extraction, write_atomic
from finrag.ingestion.pdf_loader import load_pdf


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Provenance des PDF financiers locaux et validation du jeu de données"
    )
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect-pdf", help="Inspecter la provenance et le nombre de pages d’un PDF local"
    )
    inspect.add_argument("--pdf", type=Path, required=True)
    extract = commands.add_parser(
        "extract-pages", help="Extraire chaque page physique dans un dossier privé"
    )
    extract.add_argument("--pdf", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--overwrite", action="store_true")
    chunks = commands.add_parser(
        "create-chunks", help="Créer des fenêtres de caractères limitées à chaque page"
    )
    chunks.add_argument("--pages", type=Path, required=True)
    chunks.add_argument("--output", type=Path, required=True)
    chunks.add_argument("--maximum-size", type=int, default=1250)
    chunks.add_argument("--overlap", type=int, default=0)
    chunks.add_argument("--overwrite", action="store_true")
    migration = commands.add_parser(
        "import-evaluation", help="Valider et importer le snapshot v1 privé"
    )
    for name in ("annotations", "mapping", "evidence-pdf", "output"):
        migration.add_argument(f"--{name}", type=Path, required=True)
    for name in ("corpus-manifest", "frozen", "baseline-config", "baseline-metrics"):
        migration.add_argument(f"--{name}", type=Path)
    migration.add_argument("--overwrite", action="store_true")
    validation = commands.add_parser(
        "validate-dataset", help="Vérifier empreintes, effectifs, splits et preuves associées"
    )
    validation.add_argument("--dataset", type=Path, required=True)
    corpus = commands.add_parser(
        "import-corpus", help="Vérifier les empreintes et copier les rapports autorisés en privé"
    )
    corpus.add_argument("--source-directory", type=Path, required=True)
    corpus.add_argument("--output", type=Path, default=Path("data/private/corpus/originals"))
    corpus.add_argument("--overwrite", action="store_true")
    extraction = commands.add_parser(
        "prepare-corpus", help="Extraire ou valider les caches privés de pages natives"
    )
    extraction.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("data/private/corpus/originals/corpus_manifest.json"),
    )
    extraction.add_argument("--output", type=Path, default=Path("data/private/corpus/extracted"))
    extraction.add_argument("--document", help="Traiter un seul fichier source autorisé")
    prepare = commands.add_parser(
        "prepare-lexical-chunks",
        help="Créer ou valider une configuration de découpage en caractères",
    )
    prepare.add_argument("--config", type=Path, required=True)
    _corpus_arguments(prepare)
    experiment = commands.add_parser(
        "run-lexical", help="Exécuter une expérience BM25 dev ou valider son cache"
    )
    experiment.add_argument("--config", type=Path, required=True)
    _experiment_arguments(experiment)
    experiment.add_argument("--split", choices=("dev", "test"), default="dev")
    experiment.add_argument("--allow-test", action="store_true")
    experiment.add_argument("--evaluation-reason")
    report = commands.add_parser(
        "publish-lexical-report",
        help="Valider les expériences dev terminées et produire les agrégats",
    )
    _experiment_arguments(report)
    report.add_argument("--configs", type=Path, default=Path("experiments/phase2a/configs"))
    report.add_argument(
        "--report", type=Path, default=Path("experiments/phase2a/reports/dev_results.json")
    )
    for name in ("dense-preflight", "run-dense-comparison"):
        dense = commands.add_parser(
            name, help="Précontrôle E5 local ou comparaison dev avec reprise"
        )
        dense.add_argument(
            "--model",
            type=Path,
            default=Path("models/multilingual-e5-small/614241f622f53c4eeff9890bdc4f31cfecc418b3"),
        )
        dense.add_argument("--output-root", type=Path, default=Path("artifacts/private/phase2b"))
        dense.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        dense.add_argument("--batch-size", type=int, default=4)
        if name == "run-dense-comparison":
            dense.add_argument(
                "--check-files",
                action="store_true",
                help="Valider les entrées et checkpoints, puis arrêter avant encodage",
            )
            dense.add_argument(
                "--resume",
                action="store_true",
                help="Valider et réutiliser les checkpoints publiés (comportement par défaut)",
            )
            dense.add_argument(
                "--allow-long-encoding",
                action="store_true",
                help="Exécution manuelle explicite au-delà du seuil mesuré de cinq minutes",
            )
    return root


def _corpus_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("data/private/corpus/originals/corpus_manifest.json"),
    )
    command.add_argument(
        "--extraction-root", type=Path, default=Path("data/private/corpus/extracted")
    )
    command.add_argument("--chunk-root", type=Path, default=Path("data/private/corpus/chunks"))


def _experiment_arguments(command: argparse.ArgumentParser) -> None:
    _corpus_arguments(command)
    command.add_argument("--dataset", type=Path, default=Path("data/private/evaluation/finrag_v1"))
    command.add_argument(
        "--policy", type=Path, default=Path("experiments/phase2a/selection_policy.json")
    )
    command.add_argument("--output-root", type=Path, default=Path("artifacts/private/phase2a"))


def _run_lexical(args: argparse.Namespace, config_path: Path) -> dict:
    return run_experiment(
        config_path,
        args.policy,
        args.dataset,
        args.corpus_manifest,
        args.extraction_root,
        args.chunk_root,
        args.output_root,
        Path.cwd(),
        getattr(args, "split", "dev"),
        getattr(args, "allow_test", False),
        getattr(args, "evaluation_reason", None),
    )


def run_command(args: argparse.Namespace) -> dict:
    if args.command in ("dense-preflight", "run-dense-comparison"):
        if args.batch_size <= 0:
            raise FinragError("Encoding batch size must be positive.")
        try:
            if args.command == "dense-preflight":
                from finrag.evaluation.dense_preflight import run_preflight

                return run_preflight(args.model, args.output_root, args.device, args.batch_size)
            from finrag.evaluation.run_dense_comparison import run_comparison

            return run_comparison(
                args.model,
                args.output_root,
                args.device,
                args.batch_size,
                args.allow_long_encoding,
                check_files=args.check_files,
            )
        except ImportError as error:
            raise FinragError(
                "Local dense retrieval requires the optional retrieval dependencies."
            ) from error
    if args.command == "import-corpus":
        manifest = import_corpus(args.source_directory, args.output, Path.cwd(), args.overwrite)
        return {"reports": len(manifest.reports), "hashes_verified": True}
    if args.command == "prepare-corpus":
        return {
            "documents": prepare_extraction(
                args.corpus_manifest, args.output, Path.cwd(), args.document
            )
        }
    if args.command == "prepare-lexical-chunks":
        return {
            "documents": prepare_chunks(
                args.corpus_manifest,
                args.extraction_root,
                args.chunk_root,
                load_config(args.config),
                Path.cwd(),
            )
        }
    if args.command == "run-lexical":
        result = _run_lexical(args, args.config)
        return {
            "configuration_id": result["configuration"]["configuration_id"],
            "split": result["split"],
            "metrics": result["groups"]["overall"],
            "runtime_seconds": result["runtime_seconds"],
        }
    if args.command == "publish-lexical-report":
        if any(
            not (args.output_root / name / "experiment_manifest.json").exists()
            for name in ("A", "B", "C")
        ):
            raise FinragError(
                "Public report requires three already-completed private dev experiments."
            )
        results = [_run_lexical(args, args.configs / f"{name}.json") for name in ("A", "B", "C")]
        return publish_dev_report(results, args.report)
    if args.command in ("inspect-pdf", "extract-pages"):
        manifest, pages = load_pdf(args.pdf)
        if args.command == "extract-pages":
            output = private_output(args.output, Path.cwd())
            if args.pdf.resolve().is_relative_to(output):
                raise FinragError("Extraction destination must not contain the source PDF.")
            save_page_extraction(manifest, pages, output, args.overwrite)
        return manifest.model_dump(mode="json")
    if args.command == "create-chunks":
        output = private_output(args.output, Path.cwd())
        if args.pages.resolve() == output:
            raise FinragError("Chunk output must not overwrite the input page records.")
        pages = [validate_record(PageRecord, row, "Page record") for row in read_jsonl(args.pages)]
        config = ChunkingConfig(maximum_size=args.maximum_size, overlap=args.overlap)
        chunks = chunk_pages(pages, config)
        write_atomic(
            output, jsonl_bytes([chunk.model_dump(mode="json") for chunk in chunks]), args.overwrite
        )
        return {"chunks": len(chunks), "chunking_config": config.model_dump(mode="json")}
    if args.command == "import-evaluation":
        paths = source_paths(
            args.annotations,
            args.mapping,
            args.evidence_pdf,
            args.corpus_manifest,
            args.frozen,
            args.baseline_config,
            args.baseline_metrics,
        )
        manifest = import_evaluation(paths, args.output, Path.cwd(), args.overwrite)
        return {"dataset_version": manifest.dataset_version, "counts": manifest.counts.model_dump()}
    manifest = validate_dataset(args.dataset)
    return {
        "dataset_version": manifest.dataset_version,
        "counts": manifest.counts.model_dump(),
        "valid": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run_command(args)
    except FinragError as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 2
    except ValidationError:
        print("Erreur : la configuration ne respecte pas le contrat FinRAG.", file=sys.stderr)
        return 2
    except OSError as error:
        traceback = error.__traceback__
        while traceback is not None and traceback.tb_next is not None:
            traceback = traceback.tb_next
        operation = traceback.tb_frame.f_code.co_name if traceback is not None else "filesystem"
        affected = error.filename or "chemin absent de l’exception originale"
        print(
            f"Erreur : échec de {operation} pour '{affected}' : {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
