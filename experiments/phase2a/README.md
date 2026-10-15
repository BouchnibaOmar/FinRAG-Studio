# Protocole BM25 de la Phase 2A

Configurations et politique de sélection fixées avant les expériences :

| Configuration | Maximum, caractères | Chevauchement, caractères |
|---|---:|---:|
| A | 1 000 | 100 |
| B | 2 000 | 200 |
| C | 3 000 | 300 |

Constantes : k1=1.2, b=0.75, top_k=10, tokenizer `unicode-financial-v1`.
Sélection sur les 28 cas dev répondables : Hit@5, MRR@10, Hit@1, moins de
passages, puis moins de chevauchement. Les valeurs non arrondies sont utilisées ;
un départage final par ID rend une égalité complète déterministe.
Les deux cas non répondables sont séparés. Aucun score test ne guide le choix.

## BM25 et tokenisation

Chaque passage est un document de scoring. N compte les passages, df ceux qui
contiennent le terme, L et avgL les longueurs en tokens lexicaux.

```text
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(q,c) = sum_t query_frequency(t) * idf(t) * tf(t,c) * (k1 + 1)
             / [tf(t,c) + k1 * (1 - b + b * L(c)/avgL)]
```

Variante IDF positive explicitement choisie ; les scores ne sont pas des
probabilités. Référence : Robertson et Zaragoza, *The Probabilistic Relevance
Framework: BM25 and Beyond* (2009).

Normalisation NFKC, minuscules, NFKD et retrait des accents combinés. Lettres
et chiffres restent ensemble ; suites numériques séparées par point/virgule
conservées, sans interprétation locale (`12,5` diffère de `12.5`). Pas de
lemmatisation, suppression de mots vides, traduction ou expansion d’acronymes.
Seuls les scores positifs sont retournés, triés par score décroissant puis ID.

## Provenance et évaluation

Pages physiques, blancs, offsets, configurations et empreintes sont vérifiés.
Les caches incohérents échouent sans remplacement implicite. Hit@k mesure une
page pertinente quelconque ; le rappel des pages déduplique document/page.
Une page retrouvée ne prouve pas la couverture de toutes les preuves.
Les causes d’échec non démontrées restent inconnues.

Les prédictions textuelles sont privées ; seuls agrégats, IDs et diagnostics de
pages/rangs sont publics. La référence MPNet historique a une autre segmentation.

## Commandes de référence

Depuis la racine, avec des PDF autorisés et des sorties privées disponibles.
Ces commandes d’expérimentation ne font pas partie du démarrage de l’interface.

```powershell
$ApprovedReports = Read-Host 'Dossier local contenant les PDF autorisés'
& .\.venv\Scripts\python.exe -m finrag.cli.main import-corpus `
  --source-directory $ApprovedReports
& .\.venv\Scripts\python.exe -m finrag.cli.main prepare-corpus `
  --document 'bnp_urd_2025_en.pdf'
& .\.venv\Scripts\python.exe -m finrag.cli.main prepare-corpus `
  --document 'casa_urd_2025_en.pdf'
& .\.venv\Scripts\python.exe -m finrag.cli.main prepare-corpus `
  --document 'sg_urd_2026_fy2025_fr.pdf'
foreach ($Candidate in @('A','B','C')) {
  & .\.venv\Scripts\python.exe -m finrag.cli.main prepare-lexical-chunks `
    --config "experiments/phase2a/configs/$Candidate.json"
  & .\.venv\Scripts\python.exe -m finrag.cli.main run-lexical `
    --config "experiments/phase2a/configs/$Candidate.json"
}
& .\.venv\Scripts\python.exe -m finrag.cli.main publish-lexical-report
```

Une reprise valide les caches existants ; l’import refuse l’écrasement par défaut.
Aucun nettoyage automatique. Voir [les résultats historiques](../../docs/phase2a_validation.md).
