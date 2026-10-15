# Validation de la Phase 2B

Préparation puis exécution manuelle ont eu lieu en étapes distinctes. Résultats
dev finaux sauvegardés : Hit@5 BM25 apparié **9/28**, E5 **1/28**, hybride fixe
**3/28**. Ce ne sont pas des scores de génération. Les deux questions dev non
répondables sont séparées ; aucun nouveau score test n’est utilisé.

## Protocole

`intfloat/multilingual-e5-small`, MIT, révision
`614241f622f53c4eeff9890bdc4f31cfecc418b3` : 384 dimensions, préfixes `query: `
et `passage: `, moyenne masquée, mode évaluation, normalisation float32 L2.
FAISS IndexFlatIP utilise le produit scalaire ; RRF garde constante 60 et poids égaux.

Les 7 141 fenêtres B contenaient 1 184 dépassements de 512 tokens. Les enfants
déterministes produisent **8 325 passages**, entre 19 et 512 tokens, préfixe et
tokens spéciaux inclus. Aucune troncature ; offsets et pages conservés. Les 30
questions dev comptent de 21 à 44 tokens. L’avertissement de 560 tokens provenait
du comptage d’une fenêtre B originale, pas d’un passage final encodé.

BM25 utilise les mêmes enfants que dense et hybride ; B original reste séparé.
Agrégats : `experiments/phase2b/reports/dev_results.json`. Poids, vecteurs,
checkpoints et prédictions textuelles restent privés.

## Mesures historiques

Sur MX350 de 2 Gio, batch 4 : neuf passages en 0,8537 s, soit 10,5423 passages/s.
Pic CUDA alloué : 609 072 640 octets ; réservé : 650 117 120 octets. L’estimation
avec marge de 25 % était de 986,02 s pour l’encodage restant ; elle ne garantissait
pas la durée du lot complet. Le seuil de cinq minutes a conduit au lancement manuel.

## Accès fichiers et Windows

L’ancienne erreur CLI perdait chemin et opération. Les erreurs enveloppées gardent
maintenant cause, opération, chemin et type. Le refus d’accès historique n’a pas
été reproduit : son chemin initial reste inconnu. `manual_run.log` n’est ni lu,
ni remplacé, ni validé par la commande ; un test le garde verrouillé pendant le
précontrôle. Aucun changement de permissions n’était justifié.

Le conflit Torch/FAISS a été corrigé par processus distincts. Voir
[OpenMP](phase2b_openmp_validation.md). Commande de contrôle, non exécutée par cette revue :

```powershell
& .\.venv\Scripts\python.exe -m finrag.cli.main run-dense-comparison --device cuda --batch-size 4 --resume --check-files
```

Elle s’arrête avant encodage. Un lancement complet exige explicitement
`--allow-long-encoding` à la place de `--check-files` et peut être long.
L’interface ne lance pas cette expérience. Le
[diagnostic dense](phase2b_dense_diagnostics.md) n’a démontré ni permutation des
lignes ni défaut de pooling.
