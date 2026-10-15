# Interface locale FinRAG Studio

Le [guide de démonstration](local_demo.md) décrit les écrans, modèles, coûts et
lancement. L’interface est en français et liée à `127.0.0.1`.

## Parcours

- Recherche libre : routage par entreprise puis BM25. Une traduction n’est
  réutilisée que sur égalité exacte de question originale et langue cible ;
  sinon, recherche avec la question originale, sans API implicite.
- Exemples : 28 questions dev et classements BM25/BGE sauvegardés, signalés comme
  précalculés. Aucun BGE n’est chargé.
- Aide : explications longues dans un espace séparé.

La génération utilise question originale et cinq premiers passages du classement,
indépendamment du nombre affiché. Réponses attendues et preuves annotées ne
construisent ni les résultats ni le contexte.

## Entrées locales

| Ressource | Chemin relatif à la racine |
|---|---|
| Catalogue public | `experiments/company_routing/corpus_catalog.json` |
| Passages | `artifacts/private/phase2b/model_chunks.jsonl` |
| Empreinte du corpus | `artifacts/private/phase2b/faiss_exchange/request.json` |
| Traductions corrigées et records associés | `artifacts/private/bm25_query_translation_dev/translator/cache/sets/00bf6cd63861496ea3bae30f1858e7d7.json` |
| Manifeste des candidats | `artifacts/private/bm25_candidate_reranking_dev/manifest.json` |
| Requêtes et candidats du reclasseur | `artifacts/private/bm25_candidate_reranking_dev/reranker/inputs.jsonl` |
| Correspondances et BM25 | `artifacts/private/bm25_candidate_reranking_dev/evaluation/baseline.jsonl` |
| Requêtes originales et routage | `artifacts/private/bm25_company_routing_dev/run_1/predictions_top50.jsonl` |
| Scores BGE | `artifacts/private/bge_reranker_dev/cache_cpu/*.json` |
| PDF autorisés | `data/private/corpus/originals/` |
| Catalogue de modèles | `artifacts/private/local_ui/model_catalogues/*.json` |
| Génération et diagnostics | `artifacts/private/local_generation/v2/` |

Empreintes, textes, pages et identifiants sont vérifiés avant affichage. Seuls
les champs nécessaires aux requêtes et classements sont projetés depuis les
fichiers historiques. Aucun fichier de références ou split test n’est lu.

Ces artefacts spécifiques ne sont pas distribués. Un autre corpus nécessite sa
propre préparation cohérente ; déposer un PDF ne reconstruit pas les caches.

## Citations et PDF

Les unités v2 conservent texte et positions exactes, y compris espaces insécables
et retours. Le modèle référence leurs IDs ; l’application résout texte, document
et page localement. Cette traçabilité ne garantit pas la justification financière.
Le refus concerne les passages fournis, pas l’ensemble du corpus.

Le cache inclut question, passages, empreintes, classement, modèle, prompt et
paramètres. Une modification invalide l’affichage ; une interaction visuelle ne
lance pas d’appel. Les tentatives échouées restent conservées sans retry automatique.

Le lecteur résout uniquement les IDs autorisés sous le répertoire des originaux,
refuse traversées et chemins arbitraires, vérifie empreinte et bornes physiques.
Le PDF est envoyé au navigateur local. Si le fragment de page est ignoré, numéro
physique, chemin local et téléchargement permettent une consultation locale.

Les tests `test_local_ui.py` et `test_demo_ui.py` utilisent des données synthétiques
et couvrent séparation des modes, absence d’inférence, cache exact, provenance,
PDF, fichiers absents, consentement, reruns, erreurs et refus.
