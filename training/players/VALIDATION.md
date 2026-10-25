# Validation finale — phase 7

Cette validation porte sur les contrats et l'exécution du pipeline. Les images
synthétiques et les poids aléatoires ne mesurent pas la qualité de détection sur
un match. Aucun seuil AP de recette sportive n'est affirmé. Le test final réel
reste réservé à une recette figée après sélection sur `val`.

## Installation reproductible

Depuis la racine du dépôt, Linux x86-64, Python 3.12 et `uv` (version utilisée :
0.11.7). Créer un environnement dédié, sans modifier celui de la balle :

```bash
uv venv --python 3.12 .venv-players-cpu
uv pip sync --python .venv-players-cpu/bin/python --torch-backend cpu \
  training/players/requirements-cpu.lock
uv pip check --python .venv-players-cpu/bin/python
```

L'installation télécharge des **dépendances**, pas des poids. Le lock CPU inclut
`torch==2.11.0+cpu` et `torchvision==0.26.0+cpu` ; aucun GPU ni pilote NVIDIA n'est
nécessaire. `triton` reste une dépendance transitive de `torch-hungarian`, même
dans cet environnement CPU. OpenCV requiert ses bibliothèques système (sur Ubuntu,
`libgl1` et `libglib2.0-0`/`libglib2.0-0t64`). Tk et un affichage sont nécessaires
uniquement pour les éditeurs graphiques. L'absence de Tk ou d'affichage est
rapportée comme un test GUI ignoré.

Pour CUDA 13.0, utiliser un pilote compatible et un autre environnement :

```bash
uv venv --python 3.12 .venv-players
uv pip sync --python .venv-players/bin/python --torch-backend cu130 \
  training/players/requirements-cu130.lock
uv pip check --python .venv-players/bin/python
```

| Fichier | Rôle |
| --- | --- |
| `requirements.in` | Dépendances directes et versions des API utilisées par les adaptateurs. Ce n'est pas un verrou complet. |
| `requirements-cpu.lock` | Résolution complète, directe et transitive, Python 3.12 / Linux x86-64 / PyTorch CPU. Utilisée en CI. |
| `requirements-cu130.lock` | Instantané complet de l'environnement CUDA 13.0 validé dans les phases précédentes. |
| `../../requirements.txt` | Environnement historique du projet/balle et outils DVC ; ne remplace pas les déclarations joueurs. |

Les deux locks sont spécifiques à leur plateforme et n'ont pas nécessairement
les mêmes versions transitives. Ils épinglent des versions, sans hachage des
wheels : ce n'est pas une garantie de reproduction bit à bit ni un lock
multiplateforme. Les adaptateurs refusent les versions critiques incompatibles.
Le registre embarque le lock du **runtime installé**, CPU ou CUDA, son empreinte
et l'index PyTorch correspondant, même lorsqu'un runtime CUDA infère sur CPU.
Le chargement local `mlflow.pyfunc.load_model` n'installe pas cet environnement.

Pour une mise à jour volontaire, modifier les déclarations, produire un lock
candidat, inspecter son diff puis refaire la validation dans un environnement neuf :

```bash
uv pip compile training/players/requirements.in --python-version 3.12 \
  --torch-backend cpu --output-file /tmp/players-cpu.candidate.lock
# Variante CUDA : --torch-backend cu130 et un autre fichier candidat.
```

Ne pas régénérer un lock à chaque CI. `uv pip sync` installe la résolution
retenue ; l'option de backend est nécessaire pour trouver les wheels PyTorch.
Voir les [instructions uv/PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Démo et CI sans poids externes

```bash
make validate-players PLAYERS_PYTHON=.venv-players-cpu/bin/python \
  VALIDATION_OUTPUT=runs/players/demo-cpu
```

Le dossier doit être **nouveau** ; choisir un autre nom pour une nouvelle
validation. Un résultat précédent n'est jamais effacé. Le script :

1. Exécute les contrats joueurs, les intégrations MLflow locales et les tests balle.
2. Génère localement YOLO26 Nano depuis son YAML embarqué et RF-DETR Nano avec
   `pretrain_weights=None`. Ces poids sont **aléatoires**, jamais préentraînés.
3. Pour chaque famille, crée 4 images train, 2 val et 1 test synthétiques distinctes.
   Seuls train/val sont exportés pour le parcours. Les négatifs explicites restent
   présents. Chaque image positive contient un rectangle connu, pas une prédiction
   promue en vérité terrain. Les fixtures restent sous le dossier de validation.
4. Entraîne en FP32, résolution 128, batch 2, deux epochs au total : interruption
   après l'epoch 1, corruption contrôlée du checkpoint et refus de reprise,
   restauration des octets, refus d'un changement de seed, puis reprise stricte.
5. Évalue avec COCO, vérifie les rapports et la réutilisation sans nouveau run,
   corrompt le CSV de comparaison et ne recalcule que cette étape.
6. Enregistre deux fois le candidat (même version), refuse sa promotion smoke,
   recharge via le registre et via `mlflow.pyfunc`, puis produit les boîtes et
   une vidéo annotée de quatre frames sans sidecar d'annotation vidéo.
7. Compare les deux évaluations : aucune ligne classée par défaut (smoke),
   puis deux lignes dans un même groupe de protocole avec `--include-smoke`.

Les caches de modèles sont propres à chaque validation. Les variables offline
HF/Transformers et un `sitecustomize` de test interdisent les connexions Internet
Python, y compris dans les workers. Le garde est testé dans un processus enfant ;
il n'est ni une modification globale de Python ni un pare-feu système. Le stockage
MLflow est SQLite local, les URI de tracking héritées sont neutralisées. Un
message MLflow « Downloading artifacts » désigne ici une copie depuis le magasin
local. Le script ne fait aucun DVC, aucun upload, aucun appel à un service externe.

[Le workflow CPU](../../.github/workflows/players.yml) lance cette même commande
sur push/PR affectant `training`, le Makefile ou le workflow. Il installe les
paquets avant la validation offline. Il n'utilise ni secrets ni dataset privé et
ne téléverse pas les artefacts. Les logs et `summary.json` sont affichés en CI.
L'exécution hébergée GitHub nécessite naturellement que le workflow soit committé
et poussé ; les résultats locaux ne sont pas présentés comme un succès GitHub.

Les smoke tests GPU sont **désactivés par défaut**. Activation locale explicite :

```bash
make smoke-players-gpu PLAYERS_PYTHON=.venv-players/bin/python \
  VALIDATION_OUTPUT=runs/players/demo-gpu
# Une seule famille, si nécessaire :
.venv-players/bin/python -m training.players.tests.validate \
  --output runs/players/demo-gpu-rfdetr --device cuda:0 --gpu-smoke --family rfdetr
```

Dans GitHub Actions, déclencher manuellement `Players validation` avec
`gpu_smoke=true` sur une révision de confiance. Le runner doit porter les labels
`self-hosted, linux, x64, gpu`, disposer d'un pilote compatible et de Python 3.12.
Les familles sont séquentielles et un groupe de concurrence sérialise ce job.
Aucun job GPU ne démarre sur une PR. La [syntaxe officielle](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
décrit l'entrée booléenne `workflow_dispatch`.

## Artefacts et reprise de la démo

| Chemin sous le dossier de validation | Contenu |
| --- | --- |
| `summary.json`, `players.json`, `ball.json`, `pipeline-*.json` | Statut, durées, identifiants de chaque test passé/ignoré, raisons et erreurs. |
| `*.log` | Sorties des tests et de la création de poids. |
| `weights/` | Initialisations aléatoires locales, non destinées aux matchs. |
| `yolo/`, `rfdetr/` | Sources synthétiques, recettes résolues, SQLite, parcours et `validation.json`. |
| `<famille>/pipeline/progress.json` | État atomique et historique des tentatives. |
| `<famille>/pipeline/exports/` | Paire YOLO/COCO vérifiée, générations et manifestes. |
| `<famille>/pipeline/attempts/` | Reçus, checkpoints, prédictions, CSV/JSON/HTML, courbes et exemples FP/FN. |
| `<famille>/video/` | `predictions.jsonl`, `annotated.mp4`, résultat et progression. |
| `comparison-default/`, `comparison-technical/` | Rapports communs des deux familles, sans/avec smoke. |

Les liens des artefacts MLflow sont enregistrés dans la base ; SQLite seul n'est
pas une sauvegarde des poids. Conserver aussi les magasins locaux `mlruns/` ou
`mlartifacts/` indiqués par les runs, et les sorties du parcours.

Pour revisiter une démo avec le même environnement, le même code et les mêmes
variables de backend/cache que l'exécution initiale, utiliser ses recettes
`<famille>/pipeline.json` avec `make pipeline-players`. Une modification de
l'environnement invalide volontairement les reçus. Le script de validation
vérifie déjà la relance dans un environnement identique. Il ne reprend pas sa
propre suite unittest : une nouvelle validation prend un nouveau dossier.
La reprise stricte restaure l'optimiseur et le budget ; le fine-tuning crée une
nouvelle expérience. Un checkpoint `best` conservé après reprise peut venir du
run parent si l'epoch suivante n'améliore pas le score : sa provenance doit rester
celle du run qui l'a produit, pas être renommée avec le dernier run.

## Données, annotations et restauration

Le [guide d'annotation](README.md) définit `player=0`, les XYXY en pixels source,
les états `unannotated`, `proposed`, `in_progress`, `verified`, les négatifs
vérifiés et les frontières de split par match. La préannotation reste une aide à
la revue humaine. Seules les frames explicitement vérifiées sont exportées.
Les contrats et l'audit contrôlent géométrie, empreintes et fuites de split ; ils
ne certifient pas la qualité du travail humain.

Recompter les sidecars disponibles, sans modifier `datas` :

```bash
.venv-players-cpu/bin/python -m training.players.tests.annotation_inventory \
  --source datas --output runs/players/annotations-inventory.json
```

Ce rapport compte les états déclarés après validation de schéma ; il ne vérifie
pas le contenu des médias. L'export réalise ce contrôle supplémentaire. Les
caches NBA, annotations balle/terrain et numéros de maillot ne comptent pas comme
vérité terrain de boîtes joueurs.

Pour restaurer des données référencées par DVC, utiliser un clone de la révision
voulue et sauvegarder les annotations locales avant toute restauration. Installer
DVC séparément (ou utiliser l'environnement historique qui le fournit) :

```bash
uv venv --python 3.12 .venv-players-dvc
uv pip install --python .venv-players-dvc/bin/python 'dvc[s3]==3.67.1'
# Vérifier les changements locaux avant restauration :
.venv-players-dvc/bin/dvc status datas.dvc
# Cache local déjà disponible :
.venv-players-dvc/bin/dvc checkout datas.dvc
# OU récupération explicite depuis le remote configuré :
.venv-players-dvc/bin/dvc pull datas.dvc
```

Ne pas utiliser `--force` pour écraser des annotations. Configurer les accès
locaux selon [docs/DATA.md](../../docs/DATA.md), avec un accès lecture seule pour
la restauration ; les secrets restent hors Git. Les cibles `make data-*`
mentionnées dans ce document historique ne sont pas présentes dans le Makefile
actuel : utiliser les commandes DVC ci-dessus et, pour la configuration R2,
`.venv-players-dvc/bin/python scripts/configure_r2.py` dans un terminal interactif.
Aucune restauration distante n'est nécessaire pour la démo et aucune n'est
exécutée par la CI. Aucun envoi DVC n'est automatique.

## MLflow local et distant

Les expériences sont `players-training`, `players-evaluation`, `players-registry`
et `players-registry-history`. Le modèle enregistré est `players-detector`.
`MLFLOW_TRACKING_URI` prend priorité sur `mlflow_uri` dans les recettes, pour le
tracking et le registre. Sans serveur, SQLite et les fichiers d'artefacts locaux
suffisent à toutes les commandes et aux essais de registre.

```bash
export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
make pipeline-players CONFIG=training/players/configs/pipeline.yaml
# Après adaptation des chemins de poids/recettes selon PIPELINE.md.
```

Pour un serveur **déjà administré**, renseigner son URI et fournir les secrets via
l'environnement ou le gestionnaire de secrets, jamais dans une recette commitée :

```bash
export MLFLOW_TRACKING_URI=https://mlflow.example.org
# MLFLOW_TRACKING_TOKEN est fourni séparément par votre gestionnaire de secrets.
make register-players CONFIG=training/players/configs/register.yaml
```

Ne pas mettre d'identifiants dans l'URL : la provenance enregistre les URI.
Utiliser la même destination pour entraîner, évaluer et enregistrer. Le serveur
et le client doivent pouvoir lire les artefacts ; des chemins de fichiers locaux
sur une autre machine ne suffisent pas. L'authentification dépend du serveur ;
MLflow documente notamment [le token HTTP Bearer et le stockage des artefacts](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).
Cette phase n'installe ni service web ni backend distant. Les accès distants et
l'authentification ne sont pas testés ici. Les alias restent explicites, audités
et réversibles ; le verrou applicatif local n'est pas un compare-and-swap atomique
entre plusieurs machines. Voir [PIPELINE.md](PIPELINE.md).

## Résultats et écarts

Les résultats de cette livraison et les tests ignorés sont consignés dans
[VALIDATION_RESULTS.md](VALIDATION_RESULTS.md). Les documents de phases 4 à 6
conservent leurs résultats historiques, distincts de cette validation.

Le parcours n'ajoute ni tracking d'identités, ni reconnaissance de maillots,
ni service web, ni déploiement. Restent hors validation finale sur matchs :
annotation humaine, baseline sur données réelles, campagnes multi-seeds et test
final figé. Les limites déjà livrées restent explicites : groupes d'occultation
par image, benchmark FP32 seulement, mémoire CPU échantillonnée, vidéo sans audio
et timestamps au FPS nominal, verrou de promotion distribué non atomique.
