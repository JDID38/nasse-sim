# Reproduire et prolonger

## Installer et tester

Suivre le README. Tests locaux de référence : Python 3.11, PyTorch 2.11.0 CPU, NumPy 2.4.2, Matplotlib 3.10.8, Pillow 12.2.0. La campagne GPU utilisait PyTorch 2.5.1/CUDA 12.4 sur RTX 4090. Le manifeste complet des dépendances GPU n'a pas été enregistré : nous ne prétendons pas garantir une reproduction bit à bit.

Les tests CI vérifient aussi Python 3.11 avec PyTorch 2.5.1 CPU. Les plages de `requirements.txt` ne constituent pas un verrouillage complet de l'environnement.

## Charger les résultats sans entraîner

```bash
python scripts/download_artifacts.py
python report_exit_v2.py --input artifacts/v2/campaign/evaluation_v2.json
python analyze_exit_v2_results.py --root artifacts/v2/campaign --out results/analysis_v2
```

Le fichier `docs/artifacts-v2.json` décrit l'archive publique et son empreinte. Le script refuse d'écraser un dossier de destination existant ; `--dest` permet de choisir un autre emplacement.

```bash
python evaluate_exit.py --ckpt artifacts/v2/campaign/runs_exit_0/ckpt_last.pt --B 8 --policies ia --out results/reevaluation.json
```

Cette réévaluation courte n'est pas le protocole complet : `evaluate_exit.py` n'effectue pas la sélection et les rotations du programme v2.

## Réentraîner et refaire tout le protocole

Chaque `--out` doit désigner un dossier sans checkpoint existant.

```bash
python train_exit.py --v2 --paris --seed 0 --iters 600 --B 64 --out runs/v2_seed0
python train_exit.py --v2 --paris --seed 1 --iters 600 --B 64 --out runs/v2_seed1
python train_exit.py --v2 --paris --seed 2 --iters 600 --B 64 --out runs/v2_seed2
python evaluate_exit_v2.py --checkpoints runs/v2_seed0/ckpt_last.pt runs/v2_seed1/ckpt_last.pt runs/v2_seed2/ckpt_last.pt --out results/evaluation_v2.json
python report_exit_v2.py --input results/evaluation_v2.json
```

La sélection utilise 2027 ; le test 2029. Pour un nouvel axe de recherche, réserver de nouvelles graines et documenter tout changement après consultation du test. Une nouvelle définition du succès nécessite un identifiant de protocole distinct.

## Provenance de la version publique

La version publique conserve le comportement du modèle et ajoute documentation, tests CI, chargement restreint des checkpoints et extraction sécurisée. `RotatedEnv` a été déplacé dans `exit_env.py` pour retirer la dépendance de l'évaluateur à un script d'audit local. Le JSON cartographique est réduit aux seules rues, inchangées, et aux métadonnées ; le cache utilisé par la campagne est conservé.

Les empreintes de sources dans les résultats bruts décrivent la campagne originale, antérieure à cette mise en forme. Les nouveaux rapports calculent les empreintes des sources publiques. La reconstruction des replays permet de vérifier que ces changements n'altèrent pas les transitions de l'environnement sur les trajectoires enregistrées.

Les scripts d'exploitation cloud, adresses de serveurs, notes personnelles et archives de travail ne sont pas nécessaires à la reproduction et ne sont pas publiés. Aucune commande du guide ne loue de GPU.
