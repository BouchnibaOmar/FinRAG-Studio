# Validation OpenMP sous Windows

## Cause démontrée

Versions inspectées : Torch 2.7.1+cu118, FAISS CPU 1.10.0 et NumPy 1.26.4.
Les modules chargés ont établi les provenances :

- Torch : `torch/lib/libiomp5md.dll`, runtime Intel OpenMP.
- FAISS : `faiss_cpu.libs/libomp140.x86_64-07aedf7287fc4e6e00f8128f5979c10e.dll`, runtime LLVM OpenMP.
- NumPy : OpenBLAS ; aucun de ces deux runtimes lors de son import isolé.

Les imports seuls passaient. Dans des processus frais, calcul Torch et recherche
FAISS ensemble provoquaient OpenMP Error #15, code 3, quel que soit l’ordre.
Chaque charge isolée fonctionnait. SciPy, scikit-learn et threadpoolctl étaient
absents. Les imports de tokenizers, safetensors et sentencepiece ne chargeaient
aucun de ces runtimes.

## Correction et contrôles historiques

Le parent garde Torch ; `faiss_process.py` invoque un interpréteur frais pour
`faiss_worker.py`, sans Torch. Les gardes refusent le mélange accidentel. Les
échanges sont des NPY float32 finis, normalisés, dimensionnés, sans pickle, plus
JSON empreinté. Le worker réutilise l’index valide et retourne un reçu d’intégrité.
Échecs et timeout sont explicites, sans boucle de surveillance.

Aucun `KMP_DUPLICATE_LIB_OK`, remplacement de DLL, changement global de PATH ou
suppression de cache. Aucune paire compatible dans un processus unique n’ayant
été établie, les versions fonctionnant séparément sont conservées.

E5 réel a été vérifié avec deux requêtes et passages fictifs ; vecteurs `(2, 384)`
et recherche isolée correspondaient aux produits NumPy. Les caches de 8 325
passages et 30 requêtes ont été revalidés sans réencodage. Les 4 195 fichiers
protégés sont restés identiques. Les tests ciblés historiques couvraient 31 cas
distincts ; ce compte n’est pas une nouvelle exécution.

La Phase 2B applique automatiquement l’isolation. L’interface ne charge ni
modèle local ni FAISS pour consulter les exemples.
