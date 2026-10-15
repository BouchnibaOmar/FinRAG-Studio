# Traduction contrôlée des questions dev

L’expérience compare la question originale et sa traduction dans le même oracle
BM25 documentaire. Le document vient des annotations : ce résultat diagnostique
n’est pas une mesure de routage déployable.

## Entrées et contrôle

Préparation dev seulement : 28 questions répondables, 8 325 passages inchangés,
tokenizer `unicode-financial-v1`, k1=1.2 et b=0.75. La couverture complète des
extraits du contrôle original @50 vaut 16/28 : FR→EN 6/15, FR→FR 7/7, EN→EN
3/4, EN→FR 0/2. Les 17 questions interlangues sont traduites ; les 11 autres
conservent texte, classement et métriques identiques.

Le traducteur reçoit seulement `question`, `source_language`, `target_language`.
IDs de question, documents attendus, réponses, pages, extraits et classements
restent dans le dossier d’évaluation séparé. La fidélité linguistique ne doit
pas être choisie selon les scores de recherche.

## Adaptateur et reprise

OpenRouter utilise un modèle exact configuré, la clé `OPENROUTER_API_KEY` en
environnement, un timeout et des limites d’appels/tokens. Aucun modèle choisi
par défaut. `require_parameters=true`, `allow_fallbacks=false` ; température zéro
si configurée, sans garantie de reproduction identique. L’option explicite
`--omit-temperature` modifie la configuration si nécessaire.

Une tentative réserve durablement son budget avant transport. Limites initiales :
17 appels, 40 000 tokens réservés ; 2 048 d’entrée et 256 de sortie par tentative,
soit 39 168 pour 17 appels. Il s’agit d’une politique de réservation, pas d’un
plafond de dépense ni d’un comptage exact du fournisseur. Usage et coût absents
restent inconnus. Les échecs et tentatives interrompues ne sont pas remboursés
artificiellement dans le budget et ne sont pas rejoués par `--resume`.

Les sorties vides, tronquées ou invalides sont refusées avec un motif précis.
La normalisation éventuelle de format est explicite et testée. Nombres, signes
et dates numériques sont contrôlés ; noms, unités, devises et périmètres restent
des risques de revue. Aucune validation humaine n’est inventée. Les diagnostics
privés conservent les champs utiles, sans clé ni headers d’authentification.

Les corrections locales créent des records et un ensemble immuables liés au
précédent, sans écraser réponse API, coûts ou tentatives. Le fichier d’import
contient `base_set_sha256` et `corrections` avec `input_id`, `translation`, `reason`.
Une entrée manquante est résolue depuis les entrées scellées et liée à sa
tentative échouée, sans nouvel appel ni usage API inventé.

## Commandes de référence

À lancer depuis la racine, avec sorties neuves et données privées disponibles.
Ces commandes ne sont pas exécutées pendant la revue publique. **`translate`
peut être facturé** ; `prepare`, `dry-run`, import local et évaluation sont séparés.
`pricing` consulte les métadonnées publiques ; les tarifs doivent être datés.
`OPENROUTER_MODEL` concerne cette expérience, distinct de
`OPENROUTER_GENERATION_MODEL` dans l’interface.

```powershell
& .\.venv\Scripts\python.exe -m finrag.evaluation.bm25_translation_experiment translate --experiment artifacts/private/bm25_query_translation_dev --model google/gemini-2.5-flash-lite --resume --max-calls 17 --max-total-tokens 40000
```

```powershell
& .\.venv\Scripts\python.exe -m finrag.evaluation.bm25_translation_experiment prepare --experiment artifacts/private/bm25_query_translation_dev
& .\.venv\Scripts\python.exe -m finrag.evaluation.bm25_translation_experiment dry-run --experiment artifacts/private/bm25_query_translation_dev --output artifacts/private/bm25_query_translation_dev/dry_run.json
```

```powershell
& .\.venv\Scripts\python.exe -m finrag.evaluation.bm25_translation_experiment pricing --model $env:OPENROUTER_MODEL --output artifacts/private/bm25_query_translation_dev/pricing.json
& .\.venv\Scripts\python.exe -m finrag.evaluation.bm25_translation_experiment dry-run --experiment artifacts/private/bm25_query_translation_dev --model $env:OPENROUTER_MODEL --pricing-snapshot artifacts/private/bm25_query_translation_dev/pricing.json --output artifacts/private/bm25_query_translation_dev/dry_run_priced.json
```

```powershell
$translationSet = (& .\.venv\Scripts\python.exe -m finrag.evaluation.bm25_translation_experiment translate --experiment artifacts/private/bm25_query_translation_dev --model $env:OPENROUTER_MODEL --max-calls 17 --max-total-tokens 40000 | ConvertFrom-Json)
```

```powershell
& .\.venv\Scripts\python.exe -m finrag.evaluation.bm25_translation_experiment evaluate --experiment artifacts/private/bm25_query_translation_dev --translation-set $translationSet --output artifacts/private/bm25_query_translation_dev/evaluation_run_1
```

```powershell
$correctedSet = (& .\.venv\Scripts\python.exe -m finrag.evaluation.bm25_translation_experiment import-corrections --experiment artifacts/private/bm25_query_translation_dev --translation-set $translationSet --corrections artifacts/private/bm25_query_translation_dev/local_corrections.json | ConvertFrom-Json)
& .\.venv\Scripts\python.exe -m finrag.evaluation.bm25_translation_experiment evaluate --experiment artifacts/private/bm25_query_translation_dev --translation-set $correctedSet --output artifacts/private/bm25_query_translation_dev/evaluation_run_2
```

## Mesures

Évaluation v1 : Hit@1/5/10/50, pages distinctes, couverture complète des extraits
@5/10/50, rangs, gains et régressions, langue et banque. Un seul opérande ne vaut
pas une comparaison complète. Les références alternatives ne sont pas intégrées.
L’ensemble corrigé utilisé ensuite est
`00bf6cd63861496ea3bae30f1858e7d7.json` ; son évaluation documentaire reste un
oracle. Le petit effectif et les références candidates limitent les conclusions.
