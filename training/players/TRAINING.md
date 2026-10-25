# Entraînement et modèles préentraînés — phase 4

Les adaptateurs entraînent une seule classe `player` à partir des exports vérifiés
de la phase 3. Les boîtes `referee` restent dans l'export mais sont filtrées dans
les vues privées de ces recettes joueur. Ils utilisent un seul CPU ou GPU, un run MLflow explicite par
tentative et des checkpoints complets. L'[évaluation commune de phase 5](EVALUATION.md)
compare désormais les deux familles, y compris leurs poids préentraînés sans
entraînement personnalisé. Les métriques natives ne sont pas un benchmark commun.
Le [parcours de phase 6](PIPELINE.md) relie export, entraînement et évaluation,
publie la progression des epochs/batches et réutilise cette reprise stricte.
La [validation finale](VALIDATION.md) ajoute un lock CPU, une CI offline et une
démo avec initialisation aléatoire, sans poids préentraînés externes.

## Environnement séparé

L'environnement testé est Python 3.12, PyTorch 2.11.0 / torchvision 0.26.0 CUDA
13.0, Ultralytics 8.4.39, RF-DETR 1.10.1, Lightning 2.6.6, Transformers 5.17.0 et
MLflow 3.11.1. Les contraintes directes sont dans `requirements.in` ; toutes les
distributions de l'environnement testé sont épinglées dans
`requirements-cu130.lock`. Les adaptateurs contrôlent leurs versions critiques au
démarrage. Le manifeste racine et les environnements balle restent indépendants.

Depuis la racine du dépôt, sur Linux avec un pilote compatible CUDA 13.0 :

```bash
uv venv --python 3.12 .venv-players
uv pip sync --python .venv-players/bin/python \
  training/players/requirements-cu130.lock --torch-backend cu130
uv pip check --python .venv-players/bin/python
```

Cet environnement peut aussi exécuter les modèles sur CPU. Pour une installation
sans runtime CUDA, utiliser `requirements-cpu.lock` avec `--torch-backend cpu`
dans `.venv-players-cpu`, selon [VALIDATION.md](VALIDATION.md). Pour une autre
plateforme/build PyTorch, créer un autre environnement à partir des contraintes
directes et valider les intégrations avant usage ; le lock fourni décrit la
machine de validation. Les dépendances transitives Ultralytics et supervision
demandent respectivement OpenCV standard et headless : les deux distributions
sont épinglées à 4.11.0.86. Utiliser cet environnement pour les modèles ; conserver
`.venv` pour l'éditeur graphique. Ne pas installer ces fichiers dans les
environnements balle.

## Charger des poids sans entraîner

Les poids doivent être des fichiers locaux explicites. Aucun entraînement ni
export annoté n'est nécessaire pour cette commande :

```bash
.venv-players/bin/python -m training.players.models \
  --family yolo --variant yolo26n --weights models/players/yolo26n.pt \
  --source-class 0 --image /chemin/image.png --output runs/players/prediction.json

.venv-players/bin/python -m training.players.models \
  --family rfdetr --variant rfdetr_nano --weights models/players/rf-detr-nano.pth \
  --source-class 1 --image /chemin/image.png --output runs/players/prediction-rfdetr.json
```

Omettre `--image` permet de vérifier uniquement le chargement et d'obtenir la
provenance. `--device cuda:0` active le GPU. Aucun téléchargement de poids n'est
lancé par ces commandes : les obtenir séparément auprès de leur source officielle
ou fournir les checkpoints locaux existants. Les checkpoints PyTorch doivent
provenir d'une source de confiance, leur lecture pouvant employer pickle.

Choisir explicitement la classe du checkpoint : `person=0` pour les poids YOLO
COCO testés, `person=1` pour les poids RF-DETR Nano COCO testés, `player=0` après
notre fine-tuning. Un modèle COCO « person » détecte aussi le public et les
arbitres ; renommer sa sortie ne lui apprend pas la distinction avec les joueurs.
Ces sorties restent des prédictions et ne deviennent jamais des annotations
vérifiées. Le chargement autonome ne crée pas de run d'entraînement MLflow.

L'API `training.players.models.Detector` accepte des images BGR `uint8` et renvoie
des boîtes XYXY dans les pixels source, avec `class_id=0` et `confidence`.
RF-DETR reçoit la conversion RGB appropriée. Le rapport conserve le SHA-256 des
poids, leur famille/variante, la résolution et la classe source. Les checkpoints
joueurs complets peuvent également être chargés ; les poids EMA sont utilisés
lorsqu'ils sont présents et activés.

## Recettes et modes d'initialisation

Les six recettes `configs/train_{yolo,rfdetr}_{smoke,quick,full}.yaml` sont des
points de départ validés, avec des poids locaux à renseigner :

| Recette | Epochs | Images train / val maximum | Usage |
| --- | ---: | ---: | --- |
| smoke | 2 | 4 / 2 | Vérification technique, exclue des futurs classements |
| quick | 5 | 64 / 16 | Petit essai d'une configuration |
| full | 50 | Toutes | Budget explicite à lancer volontairement |

La sélection est déterministe dans l'ordre de l'index d'export ; elle conserve
les négatifs et est enregistrée dans `selection.jsonl`. Chaque split utilisé doit
contenir au moins une boîte positive pour que les métriques soient interprétables.
Les images `test` ne sont jamais mises dans la vue d'entraînement.

Copier une recette puis adapter `weights`, `variant`, `dataset`, `device` et les
hyperparamètres souhaités. Les variantes YOLO26 n/s/m/l/x et RF-DETR
nano/small/medium/large sont acceptées ; les essais techniques portent sur les
variantes Nano. Les chemins relatifs des recettes sont résolus depuis la racine
du dépôt. Les clés inconnues, types invalides et combinaisons non prises en charge
sont refusés.

```bash
make train-players-yolo CONFIG=training/players/configs/train_yolo_smoke.yaml
make train-players-rfdetr CONFIG=training/players/configs/train_rfdetr_smoke.yaml
# Environnement alternatif : PLAYERS_PYTHON=/chemin/python
```

Les modes sont distincts :

- `mode: finetune`, `weights: /checkpoint/local`, `resume: null` : poids initiaux
  chargés, tête adaptée à `player` si nécessaire, nouvel optimiseur, nouveau
  scheduler et nouveau budget. Le SHA-256 et une copie des poids initiaux sont
  enregistrés dans MLflow.
- `mode: scratch`, `weights: null`, `resume: null` : initialisation aléatoire
  explicite, utile pour un contrôle technique.
- `mode: resume`, `resume: /run/.../last.pt` ou `.ckpt` : reprise stricte d'un
  checkpoint complet produit par cet adaptateur. `--resume` / `RESUME=...`
  sélectionne ce mode depuis la recette originale.

```bash
make train-players-yolo CONFIG=/recette/originale.yaml \
  RESUME=runs/players/<run>/yolo/weights/last.pt
make train-players-rfdetr CONFIG=/recette/originale.yaml \
  RESUME=runs/players/<run>/checkpoints/last.ckpt
```

Conserver le checkpoint `best.pt` / `best.ckpt` à côté de `last` pour préserver le
meilleur résultat antérieur. La reprise restaure le modèle d'entraînement, EMA,
l'optimiseur, le scheduler, le scaler AMP, les compteurs et l'état aléatoire.
Elle continue après la dernière epoch entièrement enregistrée ; une interruption
au milieu d'une epoch recommence cette epoch. Les sauvegardes utilisent un fichier
temporaire et un remplacement atomique, sans retirer l'état d'entraînement des
checkpoints finaux.

La reprise exige les mêmes données, sélection, architecture, budget total,
hyperparamètres, code et environnement enregistré. Les chemins opérationnels et
la destination MLflow peuvent changer. Des poids seuls, un checkpoint incompatible
ou un budget déjà terminé sont refusés. Pour changer les données, augmenter le
budget ou essayer d'autres paramètres, lancer un **nouveau fine-tuning** avec
`weights` pointant vers les poids souhaités. La reprise crée également un nouveau
run, relié à son parent par `resumed_from_run` ; elle ne réécrit pas le run parent.

## Reproductibilité et suivi

`seed` fixe les générateurs Python, NumPy et PyTorch avant la construction des
modèles. Les epochs emploient `seed + epoch` et un échantillonnage déterministe.
Cette version impose `workers: 0` et un seul device afin de maîtriser la reprise
aux frontières d'epochs. AdamW, taille de batch, résolution, précision et
augmentations sont explicites. `fp32` est le défaut ; `amp` est accepté sur CUDA.
La vérification AMP native YOLO, qui peut télécharger un autre modèle, est
désactivée au profit de ce choix explicite et des essais d'intégration.
RF-DETR expose `rfdetr.amp_init_scale` (128 par défaut) : une échelle initiale
modérée permet aux essais courts d'effectuer de vraies mises à jour. Le scaler
continue ensuite son adaptation automatique et son état est restauré à la reprise.

`deterministic: true` configure PyTorch en mode avertissement : certaines
opérations CUDA de RF-DETR, notamment le gradient de `grid_sample`, restent
non déterministes. Une reprise stricte restaure l'état complet ; elle ne promet
pas une égalité bit à bit sur tous les GPU. Les versions du runtime, le matériel
et la précision font partie de la provenance et du contrat de reprise.

Le suivi réutilise `training/common/tracking.py`, avec une expérience
`players-training` distincte. `MLFLOW_TRACKING_URI` prend priorité sur la recette :

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 make train-players-yolo CONFIG=/recette.yaml
```

Les intégrations MLflow automatiques des frameworks sont désactivées pour éviter
les runs doublons. Chaque tentative trace configuration résolue, manifeste
dataset, `dataset_id` / `export_id` / `selection_id`, empreintes du code, état Git,
distributions Python, GPU/runtime, poids initiaux, métriques par epoch, temps,
configurations natives et checkpoints avec SHA-256. Les données des frameworks
sont préparées sous le run ; leurs caches n'altèrent pas les exports originaux.

Sous `runs/players/<run_id>/`, consulter notamment `config.resolved.yaml`,
`environment.json`, `manifest.json`, `initialization.json`, `selection.jsonl`,
`framework.resolved.yaml`, `metrics.jsonl`, `checkpoints.json` et `summary.json`.
Un échec conserve les fichiers disponibles et produit `error.json` ; un arrêt
géré est marqué `KILLED` dans MLflow. Un arrêt brutal tel que `SIGKILL` ne peut pas
exécuter le code de finalisation. DVC n'est ni actualisé ni envoyé par l'entraînement.

## Essais courts

```bash
make test-players
make test-ball
PLAYERS_MLFLOW_TESTS=1 .venv-players/bin/python -m unittest training.players.tests.test_tracking -v

PLAYERS_TRAINING_TESTS=yolo PLAYERS_TEST_WEIGHTS=/chemin/yolo26n.pt \
  PLAYERS_TEST_DEVICE=cpu make test-players-training
PLAYERS_TRAINING_TESTS=rfdetr PLAYERS_TEST_WEIGHTS=/chemin/rf-detr-nano.pth \
  PLAYERS_TEST_DEVICE=cuda:0 make test-players-training
```

Ces intégrations créent leurs images et annotations synthétiques, exportent une
petite paire YOLO/COCO et utilisent une base MLflow locale isolée. Elles comparent
deux epochs continues à une epoch interrompue puis reprise, vérifient les runs,
rechargent les poids, lancent une epoch de nouveau fine-tuning et contrôlent que
les exports sont inchangés. Elles exigent une vraie mise à jour de l'optimiseur,
comparent ses états, ceux du scheduler et d'EMA, et vérifient les statuts MLflow.
Aucun cache de
prédictions existant n'est transformé en vérité terrain. `PLAYERS_TEST_OUTPUT`
permet de conserver leurs artefacts ; `PLAYERS_TEST_PRECISION=amp` permet de tester
la précision mixte sur GPU.

Pour tester manuellement l'interruption, le CLI accepte `--stop-after-epoch 1`
avec une recette ayant au moins deux epochs. Il s'arrête après une sauvegarde
complète, avec un statut `KILLED`, afin de vérifier ensuite `--resume`.

### Résultats techniques du 23 septembre 2026

Les essais portent sur YOLO26 Nano et RF-DETR Nano, avec quatre images train et
deux images val synthétiques, sur CPU et une RTX 5080. Chaque intégration exécute
deux epochs continues, une epoch interrompue puis reprise, et une epoch de
fine-tuning indépendant. Aucun entraînement long sur les données du projet.

| Famille | Device / précision | Écart maximal des poids, continu contre reprise |
| --- | --- | ---: |
| YOLO26 Nano | CPU / FP32 | 0 |
| YOLO26 Nano | RTX 5080 / AMP | 0 |
| RF-DETR Nano | CPU / FP32 | 0 |
| RF-DETR Nano | RTX 5080 / FP32 | 0,000150 |
| RF-DETR Nano | RTX 5080 / AMP | 0,000230 |

Les écarts CUDA RF-DETR sont cohérents avec les opérations non déterministes
signalées par PyTorch ; ces résultats ne constituent pas une garantie générale
d'égalité des poids sur GPU. Les contrôles du chargement préentraîné, du
rechargement des checkpoints, des états d'optimisation et des quatre runs MLflow
distincts ont passé dans chaque intégration. Les quatre tests dédiés au tracker
MLflow ont également passé.

Suite CPU joueurs : 67 tests réussis, 7 intégrations optionnelles désactivées.
Non-régression balle : 56 tests réussis, 13 intégrations optionnelles désactivées.
Les artefacts locaux et identifiants des runs sont conservés dans
`runs/players/phase4-validation/validation.summary.json` et les sous-dossiers
`integrations/`. Ces résultats vérifient le fonctionnement du pipeline ; les
métriques sur ces fixtures ne mesurent pas la qualité d'un détecteur de joueurs.

Références des API examinées : [entraînement Ultralytics](https://docs.ultralytics.com/modes/train/),
[API publique RF-DETR/Lightning](https://rfdetr.roboflow.com/latest/learn/train/customization/).
Les adaptateurs sont validés contre les versions épinglées ci-dessus, même si la
documentation amont évolue.
