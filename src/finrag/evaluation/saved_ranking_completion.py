import argparse
from collections import defaultdict
from pathlib import Path

from finrag.config import FinragError, private_output
from finrag.domain.dense import DenseHit
from finrag.evaluation.dev_routing_diagnostics import (
    Hit,
    PageKey,
    distinct_pages,
    first_reference_rank,
    format_fraction,
    fraction,
    load_predictions,
    page_key,
    validate_ranking,
)
from finrag.evaluation.split_guard import load_dev_questions
from finrag.ingestion.hashing import hash_file
from finrag.ingestion.manifest import json_bytes, read_json, write_atomic


def page_positions(hits: list[Hit], target: PageKey) -> dict:
    filtered = [hit for hit in hits if hit["document_id"] == target[0]]
    expected = {target}
    return {
        "physical_page": target[1],
        "present": any(page_key(hit) == target for hit in hits),
        "segment_ranks": [hit["rank"] for hit in hits if page_key(hit) == target],
        "first_segment_rank": first_reference_rank(hits, expected),
        "global_distinct_page_rank": first_reference_rank(distinct_pages(hits), expected),
        "filtered_segment_rank": first_reference_rank(filtered, expected),
        "filtered_distinct_page_rank": first_reference_rank(distinct_pages(filtered), expected),
    }


def filtered_observation(hits: list[Hit], document: str, reference_pages: list[int]) -> dict:
    expected = {(document, page) for page in reference_pages}
    filtered = [hit for hit in hits if hit["document_id"] == document]
    pages = distinct_pages(filtered)
    rank = first_reference_rank(pages, expected)
    result = {
        "saved_global_segments": len(hits),
        "available_document_segments": len(filtered),
        "available_document_pages": len(pages),
        "first_reference_segment_rank": first_reference_rank(hits, expected),
        "first_reference_global_page_rank": first_reference_rank(distinct_pages(hits), expected),
        "first_reference_filtered_segment_rank": first_reference_rank(filtered, expected),
        "first_reference_filtered_page_rank": rank,
        "reference_pages": [page_positions(hits, target) for target in sorted(expected)],
        "saved_ranking_is_nonexhaustive": True,
    }
    for k in (5, 10):
        observed = rank is not None and rank <= k
        incomplete = len(pages) < k
        result[f"observed_hit{k}"] = observed
        result[f"incomplete_for_hit{k}"] = incomplete
        result[f"observation_state{k}"] = (
            "observed_positive"
            if observed
            else "incomplete_no_reference_observed"
            if incomplete
            else "complete_prefix_negative"
        )
    return result


def observation_summary(rows: list[dict]) -> dict:
    result = {}
    for k in (5, 10):
        eligible = [row for row in rows if not row[f"incomplete_for_hit{k}"]]
        result[f"hit{k}"] = {
            "observed_all_questions": fraction([row[f"observed_hit{k}"] for row in rows]),
            "sufficient_saved_pages_only": fraction([row[f"observed_hit{k}"] for row in eligible]),
            "eligible_question_ids": [row["question_id"] for row in eligible],
            "incomplete_question_ids": [row["question_id"] for row in rows if row not in eligible],
            "observed_positive_incomplete": sum(
                row[f"observed_hit{k}"] and row[f"incomplete_for_hit{k}"] for row in rows
            ),
        }
    return result


def proposed_set_observation(hits: list[Hit], document: str, pages: list[int]) -> dict:
    positions = [page_positions(hits, (document, page)) for page in pages]
    return {
        "proposed_pages": pages,
        "page_positions": positions,
        "all_proposed_pages_observed": all(row["present"] for row in positions),
        "observed_page_count": sum(row["present"] for row in positions),
        "counted_as_metric_success": False,
        "proposal_validation_status": "unvalidated",
    }


def operand_excerpt_coverage(
    hits: list[Hit], document: str, source_rows: dict[int, tuple[str, str]]
) -> dict:
    observed = []
    for page, (scope, excerpt) in source_rows.items():
        ranks = [
            hit["rank"]
            for hit in hits
            if page_key(hit) == (document, page)
            and " ".join(excerpt.split()) in " ".join(hit["text"].split())
        ]
        if ranks:
            observed.append({"scope": scope, "physical_page": page, "segment_ranks": ranks})
    scopes = sorted({row["scope"] for row in observed})
    return {
        "operand_excerpts_observed": observed,
        "operand_scopes_observed": scopes,
        "both_proposed_operand_excerpts_observed": scopes == ["group", "sa"],
        "financial_validation_status": "unvalidated proposal; excerpt presence only",
    }


def inspect_previous_report(previous: dict) -> dict:
    return {
        "already_contains_filtered_hit5_scores": True,
        "previous_scores": {
            method: analysis["groups"]["overall"]["oracle_page_hit5"]
            for method, analysis in previous["analyses"].items()
        },
        "missing_views": [
            "observed filtered page Hit@5/Hit@10 over all 28 answerable questions",
            "separate sufficient-depth cohorts for Hit@5 and Hit@10",
            "dense top50 reference-page rank list and alternative-page positions",
        ],
    }


def render_completion(report: dict) -> str:
    lines = [
        "# Étape A — complément sur les classements sauvegardés",
        "",
        "Dev uniquement : 28 répondables ; les deux non-répondables sont exclus et listés "
        "dans le JSON. Aucun accès au test, aucune recherche ni inférence nouvelle.",
        "",
        "## Ce que contenait déjà le rapport",
        "",
        "Le rapport précédent contenait déjà les Hit@5 après filtrage et déduplication, "
        "sur les listes ayant cinq pages : BM25 9/12, dense top10 1/19, hybride 3/13, "
        "dense top50 1/25. Il ne contenait pas seulement des compteurs de profondeur.",
        "",
        "## Observations après filtrage puis déduplication",
        "",
        "Filtre sur le document v1 attendu, puis déduplication par document/page physique "
        "en première apparition. Hit signifie au moins une page v1, même pour une comparaison. "
        "Les observations sur 28 questions ne traitent pas les absences tronquées comme "
        "des échecs exhaustifs. Les listes suffisamment profondes sont présentées séparément.",
        "",
        "| Source sauvegardée | Hit5 observé /28 | Hit10 observé /28 | "
        "Hit5 listes ≥5 pages | Hit10 listes ≥10 pages | Incomplètes pour 5 / 10 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for source, analysis in report["analyses"].items():
        five, ten = analysis["summary"]["hit5"], analysis["summary"]["hit10"]
        lines.append(
            f"| {source} | {format_fraction(five['observed_all_questions'])} | "
            f"{format_fraction(ten['observed_all_questions'])} | "
            f"{format_fraction(five['sufficient_saved_pages_only'])} | "
            f"{format_fraction(ten['sufficient_saved_pages_only'])} | "
            f"{len(five['incomplete_question_ids'])}/{len(ten['incomplete_question_ids'])} |"
        )
    lines.extend(
        [
            "",
            "Ces sous-ensembles sont différents : leurs taux ne sont pas directement "
            "comparables entre méthodes ou entre Hit5 et Hit10. Les IDs admissibles et les "
            "succès observés sur listes incomplètes sont conservés dans le JSON. Tous les "
            "classements restent des préfixes globaux sauvegardés ; cinq ou dix pages "
            "disponibles ne rendent pas l'oracle exhaustif.",
            "",
            "## Pages v1 dans le top50 dense : 28 questions",
            "",
            f"Présence d'au moins une page v1 : "
            f"**{format_fraction(report['dense_top50_presence'])}**.",
            "",
            "S = premier rang de segment global ; P = rang de page distincte global ; "
            "FS = rang de segment filtré sur le document ; FP = rang de page distincte filtré. "
            "`—` signifie non observé, pas absent du document. Le rang d'une page "
            "ne garantit pas que le segment contient l'extrait attendu.",
            "",
            "| ID | Présente | S | P | FS | FP | Pages filtrées | Incomplet 5 / 10 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["analyses"]["dense_top50"]["questions"]:
        ranks = [
            row[key] if row[key] is not None else "—"
            for key in (
                "first_reference_segment_rank",
                "first_reference_global_page_rank",
                "first_reference_filtered_segment_rank",
                "first_reference_filtered_page_rank",
            )
        ]
        lines.append(
            f"| {row['question_id']} | {ranks[0] != '—'} | "
            f"{' | '.join(map(str, ranks))} | {row['available_document_pages']} | "
            f"{row['incomplete_for_hit5']} / {row['incomplete_for_hit10']} |"
        )
    lines.extend(
        [
            "",
            "Liste des premiers rangs de segment des succès v1, dans l'ordre dev : "
            f"`{report['dense_success_segment_ranks']}`. Le JSON inclut aussi tous les rangs "
            "de segments pour chaque page v1, pas seulement le premier.",
            "",
            "## Présence des propositions alternatives — aucun succès ajouté",
            "",
            "Uniquement les ensembles déjà proposés dans le paquet privé sont examinés. "
            "Leurs pages restent non validées. La page 685 est le contexte de périmètre "
            "du tableau 684, pas une seconde valeur du coût du risque. Pour casa_09, les "
            "autres sections corroborent le fait mais ne sont pas la section littérale "
            "chiffres clés.",
            "",
            "Les tuples de rangs ci-dessous suivent S/P/FS/FP. Les pages non observées sont "
            "indiquées explicitement. Pour casa_11, les deux ensembles proposés sont "
            "484(S.A.) + 501(Groupe), et 731(S.A.) + 928(Groupe).",
            "",
            "| Question | Source | Pages proposées présentes : S/P/FS/FP | Non observées | "
            "Ensembles proposés observés, non validés | Fragments d'opérandes casa_11 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    grouped_alternatives = defaultdict(list)
    for alternative in report["alternative_observations"]:
        grouped_alternatives[(alternative["question_id"], alternative["source"])].append(
            alternative
        )
    for (question_id, source), alternatives in grouped_alternatives.items():
        present, absent = [], []
        positions = {
            position["physical_page"]: position
            for alternative in alternatives
            for position in alternative["page_positions"]
        }
        for position in positions.values():
            if position["present"]:
                ranks = [
                    position[key]
                    for key in (
                        "first_segment_rank",
                        "global_distinct_page_rank",
                        "filtered_segment_rank",
                        "filtered_distinct_page_rank",
                    )
                ]
                present.append(f"{position['physical_page']}: {'/'.join(map(str, ranks))}")
            else:
                absent.append(str(position["physical_page"]))
        complete_sets = [
            "+".join(map(str, alternative["proposed_pages"]))
            for alternative in alternatives
            if alternative["all_proposed_pages_observed"]
        ]
        operands = [
            "+".join(map(str, alternative["proposed_pages"]))
            + ": "
            + (",".join(alternative["operand_scopes_observed"]) or "aucun")
            for alternative in alternatives
            if "operand_scopes_observed" in alternative
        ]
        lines.append(
            f"| {question_id} | {source} | "
            f"{'; '.join(present) or '—'} | {', '.join(absent) or '—'} | "
            f"{'; '.join(complete_sets) or 'aucun'} | {'; '.join(operands) or '—'} |"
        )
    lines.extend(
        [
            "",
            "Pour casa_11, les fragments sont les lignes exactes déjà présentes dans les "
            "pages du paquet de revue (espaces normalisés). Une page peut apparaître sans "
            "que son segment contienne cette ligne. Un seul opérande reste partiel. "
            "Même un ensemble "
            "complet observé ne valide pas automatiquement les propositions et ne change "
            "aucun score. Les fichiers existants, références v1 et moteur sont préservés.",
            "",
            "## Limites",
            "",
            "BM25/hybride : top10 uniquement ; dense : top10 et top50. Une absence dans ces "
            "préfixes reste une observation limitée. Aucune recherche réellement restreinte "
            "au document n'a été exécutée. Les rangs portent sur les pages ; seul casa_11 "
            "ajoute un contrôle de présence des lignes d'opérandes proposées. Aucun effet "
            "causal de la langue ou du modèle "
            "n'est établi. Les observations de chaque question pour toutes les sources sont "
            "dans completion.json ; les matrices d'embeddings ne sont pas lues.",
            "",
            "Hashes avant/après des entrées identiques. Les Hit@5 historiques restent "
            "BM25 9/28, dense 1/28 et hybride 3/28.",
            "",
        ]
    )
    return "\n".join(lines)


def complete_saved_rankings(project_root: Path, output: Path) -> dict:
    output = private_output(output, project_root)
    if output.exists():
        raise FinragError("Completion requires a new directory; existing files are preserved.")
    dataset = project_root / "data/private/evaluation/finrag_v1"
    phase = project_root / "artifacts/private/phase2b"
    old_root = project_root / "artifacts/private/dev_routing_step_a_validated"
    proposals_path = project_root / (
        "artifacts/private/reference_reviews/phase2b_four_dev/proposed_corrections.json"
    )
    paths = [
        dataset / "dev.jsonl",
        dataset / "dataset_manifest.json",
        old_root / "diagnostic.json",
        old_root / "rapport.md",
        proposals_path,
        phase / "faiss_exchange/results.json",
        phase / "faiss_exchange/request.json",
        phase / "queries/cache_manifest.json",
    ]
    paths += [phase / method / "predictions.jsonl" for method in ("bm25", "dense", "hybrid")]
    operand_sources = {
        page: project_root
        / (
            "artifacts/private/reference_reviews/phase2b_four_dev/source_text/"
            f"finrag_v1_casa_11/physical_{page:04}.txt"
        )
        for page in (484, 501, 731, 928)
    }
    paths += list(operand_sources.values())
    before = {str(path): hash_file(path) for path in paths}
    previous = read_json(old_root / "diagnostic.json")
    for path in paths:
        previous_hash = previous["input_hashes_after"].get(str(path.resolve()))
        if previous_hash is not None and previous_hash != before[str(path)]:
            raise FinragError(f"Saved input changed since Step A: {path}.")
    questions = load_dev_questions(dataset)
    query_manifest = read_json(phase / "queries/cache_manifest.json")
    if query_manifest["fingerprint"]["dev_sha256"] != before[str(dataset / "dev.jsonl")]:
        raise FinragError("Saved dense query order does not match frozen dev.")
    saved = {
        f"{method}_top10": load_predictions(phase / method / "predictions.jsonl", questions, True)
        for method in ("bm25", "dense", "hybrid")
    }
    for source, predictions in saved.items():
        if any(row["method"] != source.removesuffix("_top10") for row in predictions):
            raise FinragError("Saved method differs from its directory.")
    exchange = read_json(phase / "faiss_exchange/results.json")
    if exchange["request_sha256"] != before[str(phase / "faiss_exchange/request.json")]:
        raise FinragError("Saved dense results refer to another request.")
    if len(exchange["hits"]) != len(questions) or exchange["query_count"] != len(questions):
        raise FinragError("Saved dense results must match all dev rows.")
    saved["dense_top50"] = []
    for prediction, hits in zip(saved["dense_top10"], exchange["hits"], strict=True):
        hits = [DenseHit.model_validate(hit).model_dump(mode="json") for hit in hits]
        validate_ranking(hits)
        if len(hits) != 50 or hits[: len(prediction["retrieved"])] != prediction["retrieved"]:
            raise FinragError("Saved dense top50 must reproduce the final top10 exactly.")
        saved["dense_top50"].append({**prediction, "retrieved": hits})
    analyses, excluded = {}, []
    for source, predictions in saved.items():
        rows = []
        for question, prediction in zip(questions, predictions, strict=True):
            if not question.answerable:
                if source == "bm25_top10":
                    excluded.append(question.question_id)
                continue
            rows.append(
                {
                    "question_id": question.question_id,
                    **filtered_observation(
                        prediction["retrieved"], question.document_id, question.relevant_pages
                    ),
                }
            )
        if len(rows) != 28:
            raise FinragError("Completion requires all 28 answerable dev questions.")
        analyses[source] = {"questions": rows, "summary": observation_summary(rows)}
        if source.endswith("_top10"):
            historical_count = sum(
                row["first_reference_segment_rank"] is not None
                and row["first_reference_segment_rank"] <= 5
                for row in rows
            )
            method = source.removesuffix("_top10")
            if historical_count != previous["historical_verification"][method]["numerator"]:
                raise FinragError("Frozen historical Hit@5 differs from saved observations.")
    by_question = {question.question_id: question for question in questions}
    operand_rows = {}
    for page, path in operand_sources.items():
        text = path.read_text(encoding="utf-8")
        anchor = (
            "Net income Group share during the period" if page == 731 else "NET INCOME GROUP SHARE"
        )
        source_row = next((line for line in text.split("\n") if line.startswith(anchor)), None)
        if source_row is None:
            raise FinragError(f"Proposed operand source row not found on physical page {page}.")
        operand_rows[page] = ("sa" if page in (484, 731) else "group", source_row)
    alternatives = []
    for proposal in read_json(proposals_path):
        question = by_question.get(proposal["question_id"])
        if question is None or not question.answerable:
            raise FinragError("Alternative proposals must refer to answerable dev questions.")
        if proposal["current_reference"]["relevant_pages"] != question.relevant_pages:
            raise FinragError("Proposal baseline differs from current v1 pages.")
        for source, predictions in saved.items():
            prediction = next(
                row for row in predictions if row["question_id"] == question.question_id
            )
            for pages in proposal["alternative_page_sets"]:
                observation = {
                    "question_id": question.question_id,
                    "source": source,
                    **proposed_set_observation(
                        prediction["retrieved"], question.document_id, pages
                    ),
                }
                if question.question_id == "finrag_v1_casa_11":
                    observation.update(
                        operand_excerpt_coverage(
                            prediction["retrieved"],
                            question.document_id,
                            {page: operand_rows[page] for page in pages},
                        )
                    )
                alternatives.append(observation)
    dense_rows = analyses["dense_top50"]["questions"]
    after = {str(path): hash_file(path) for path in paths}
    if before != after:
        raise FinragError("An existing input changed during completion.")
    report = {
        "split": "dev",
        "test_read": False,
        "new_search": False,
        "new_encoding": False,
        "previous_report_inspection": inspect_previous_report(previous),
        "analyses": analyses,
        "unanswerable_excluded": excluded,
        "dense_top50_presence": fraction(
            [row["first_reference_segment_rank"] is not None for row in dense_rows]
        ),
        "dense_success_segment_ranks": [
            row["first_reference_segment_rank"]
            for row in dense_rows
            if row["first_reference_segment_rank"] is not None
        ],
        "alternative_observations": alternatives,
        "input_hashes_before": before,
        "input_hashes_after": after,
        "all_existing_read_inputs_unchanged": True,
    }
    write_atomic(output / "completion.json", json_bytes(report))
    write_atomic(output / "rapport.md", render_completion(report).encode("utf-8"))
    return {
        "report": str(output / "rapport.md"),
        "dense_top50_presence": report["dense_top50_presence"],
        "input_files_preserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete dev saved-ranking observations; no search"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(complete_saved_rankings(Path.cwd(), args.output))


if __name__ == "__main__":
    main()
