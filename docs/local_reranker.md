# Reclassement local expérimental BGE

L’adaptateur conserve les 50 candidats `cached_translation` de chaque question
dev répondable et la requête effectivement recherchée dans les entrées scellées.
Il ne remplace pas la question traduite par l’originale, ne lance pas de recherche
et ne lit pas les annotations pendant le scoring. OpenRouter reste indépendant.

## Modèle figé

`BAAI/bge-reranker-v2-m3`, révision `324cc40576b08b305b9c65a867c26c173a477ae2`.
La [fiche de cette révision](https://huggingface.co/BAAI/bge-reranker-v2-m3/blob/324cc40576b08b305b9c65a867c26c173a477ae2/README.md)
indique Apache-2.0. Poids `model.safetensors` d’environ 2,27 Go, SHA-256
`d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`.
Les termes du modèle s’appliquent séparément des droits sur le corpus.

La limite retenue est de 8 192 positions pour **question + passage + tokens
spéciaux** ; XLM-R déclare 8 194 positions avec décalage de padding. Les passages
compatibles E5 ne suffisent donc pas à certifier les paires BGE.

Les versions sont celles de l’extra `retrieval` de `pyproject.toml`. L’adaptateur
utilise directement `AutoModelForSequenceClassification`, sans FlagEmbedding,
sentence-transformers ou code distant. Chargements locaux uniquement,
`trust_remote_code=False`. Le processus d’inférence ne charge pas FAISS.

## Précontrôle, matériel et cache

Le tokenizer compte les 1 400 paires complètes sans troncature avant chargement
des poids. Tout dépassement bloque explicitement. Les scores doivent être finis,
triés par ordre décroissant et départagés déterministement ; aucun score de
remplacement n’est inventé.

L’essai matériel utilise batch 1 : échauffement puis neuf paires courtes, médianes
et longues choisies sans scores ni annotations. Il enregistre chargement, durées,
débit et mémoire disponibles, avec synchronisation CUDA si concernée. Une
extrapolation n’est pas une garantie de durée. Le GPU historique MX350 de 2 Gio
ne suffit pas à promettre le fonctionnement des poids FP32. Aucun basculement
CPU/GPU silencieux ; float16 est une configuration séparée.

Cache et erreurs conservent configuration, empreintes et contexte technique sans
texte documentaire ni secret dans les journaux techniques. Les causes originales
restent chaînées ; les fichiers existants ne sont pas remplacés.

## Commandes explicites

Ces commandes nécessitent les entrées privées. Elles sont des procédures de
référence, **non exécutées pendant la revue publique**. Le téléchargement est
une action séparée. Choisir des sorties neuves pour une autre exécution.

### Téléchargement et scellement

```powershell
& .\.venv\Scripts\huggingface-cli.exe download BAAI/bge-reranker-v2-m3 `
  --revision 324cc40576b08b305b9c65a867c26c173a477ae2 `
  --include config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json sentencepiece.bpe.model README.md `
  --max-workers 1 `
  --local-dir data/private/models/bge-reranker-v2-m3-324cc405

& .\.venv\Scripts\python.exe -m finrag.retrieval.local_reranker_assets `
  --model-directory data/private/models/bge-reranker-v2-m3-324cc405
```

### Précontrôle sans poids

```powershell
& .\.venv\Scripts\python.exe -m finrag.evaluation.local_reranker_experiment preflight `
  --model-directory data/private/models/bge-reranker-v2-m3-324cc405 `
  --device cpu --dtype float32 `
  --output artifacts/private/bge_reranker_dev/preflight_cpu.json
```

### Essai matériel court

```powershell
& .\.venv\Scripts\python.exe -m finrag.evaluation.local_reranker_experiment smoke `
  --model-directory data/private/models/bge-reranker-v2-m3-324cc405 `
  --device cpu --dtype float32 `
  --output artifacts/private/bge_reranker_dev/smoke_cpu.json
```

### Lot complet explicite

```powershell
& .\.venv\Scripts\python.exe -m finrag.evaluation.local_reranker_experiment run `
  --model-directory data/private/models/bge-reranker-v2-m3-324cc405 `
  --device cpu --dtype float32 `
  --smoke-report artifacts/private/bge_reranker_dev/smoke_cpu.json `
  --cache artifacts/private/bge_reranker_dev/cache_cpu `
  --output artifacts/private/bge_reranker_dev/run_cpu.json
```

### Évaluation séparée

```powershell
& .\.venv\Scripts\python.exe -m finrag.evaluation.local_reranker_experiment evaluate `
  --model-directory data/private/models/bge-reranker-v2-m3-324cc405 `
  --device cpu --dtype float32 `
  --cache artifacts/private/bge_reranker_dev/cache_cpu `
  --output artifacts/private/bge_reranker_dev/evaluation_cpu.json
```

## Résultats et interprétation

Le lot CPU sauvegardé contient 28 succès. Sur les mêmes 28 questions : BM25
Hit@1 5/28 et couverture complète @5 11/28 ; BGE Hit@1 9/28 et couverture
complète @5 16/28. Les comparaisons Mistral sont séparées, appariées aux mêmes
12 sorties valides, sans extrapolation à 28. Les échecs restent dans les
dénominateurs opérationnels. La présence d’un extrait n’est pas une validation
de suffisance financière. Les références v1 et leurs statuts restent inchangés.

Les tests emploient explicitement des doublures de tokenizer, scoreur et actifs ;
leurs durées ne sont pas des mesures BGE. Voir `retrieval/local_reranker.py`,
`local_reranker_assets.py` et `evaluation/local_reranker_experiment.py`.
