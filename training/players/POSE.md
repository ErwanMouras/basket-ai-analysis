# Squelettes des joueurs

`pose-players` exécute détection → tracking → RTMPose-M → vidéo annotée et JSONL.
Le modèle estime 17 points corporels COCO en 2D dans chaque boîte retenue, en
conservant son `track_id`. Le module [pose.py](pose.py) reproduit le cadrage,
la normalisation et le décodage utilisés dans NBA avec RTMPose-M/rtmlib 0.0.15.
Il fonctionne ici avec ONNX Runtime, sans importer NBA ni installer rtmlib.
Le contrat de boîtes est commun aux détecteurs YOLO et RF-DETR.

## Installation et lancement

Utiliser les environnements joueurs de [TRAINING.md](TRAINING.md).
Les locks actualisés ajoutent ONNX Runtime 1.27.0 et FlatBuffers 25.12.19.
Installer **un seul** runtime dans chaque environnement : `onnxruntime` pour CPU,
`onnxruntime-gpu` pour CUDA 13. Les entrées correspondantes sont
`requirements-cpu.in` et `requirements-cu130.in` ; `requirements.in` contient
les dépendances communes. Ne pas installer les deux distributions ensemble.

```bash
# Environnement CPU existant : synchroniser ses dépendances.
uv pip sync --python .venv-players-cpu/bin/python --torch-backend cpu \
  training/players/requirements-cpu.lock
make setup-players-pose PLAYERS_PYTHON=.venv-players-cpu/bin/python
make pose-players PLAYERS_PYTHON=.venv-players-cpu/bin/python \
  VIDEO=datas/train/HOU_LAL/G1/video.mp4 \
  CHECKPOINT=/chemin/vers/yolo26n.pt \
  OUTPUT=runs/players/match-pose MAX_FRAMES=300
```

`setup-players-pose` télécharge explicitement le modèle officiel et vérifie son
SHA-256 avant installation atomique dans `models/players/rtmpose-m.onnx`.
Un modèle identique déjà présent est réutilisé. Un fichier différent est refusé.
L'inférence ne télécharge aucun poids. Pour importer le même ONNX depuis un
cache local, ajouter `POSE_SOURCE=/chemin/vers/rtmpose-m.onnx` à la commande setup.
`POSE_WEIGHTS=/chemin/local.onnx` choisit son emplacement, au setup comme au run.
Seul le modèle épinglé décrit ci-dessous est accepté.

Le dossier de sortie doit être nouveau. Le défaut utilise YOLO26 Nano CPU 640,
BoT-SORT et RTMPose CPU. `TRACKER=bytetrack` change le suivi ;
`TRACKER=none` permet une pose sans suivi, avec `track_id: null`.
`POSE=1` active aussi les squelettes dans `make predict-players` ou `track-players` ;
`POSE=0` les désactive. Les anciennes recettes gardent la pose désactivée.

Pour CUDA, synchroniser `requirements-cu130.lock` avec `--torch-backend cu130`
dans `.venv-players`, puis utiliser cet environnement. `POSE_DEVICE=cuda:0`
choisit le GPU de pose indépendamment du détecteur. Pour mettre **les deux**
sur GPU, copier [predict_video.yaml](configs/predict_video.yaml), régler son
`device: cuda:0`, puis lancer :

```bash
make pose-players CONFIG=/chemin/recette-gpu.yaml POSE_DEVICE=cuda:0 \
  VIDEO=/videos/match.mp4 CHECKPOINT=/poids/yolo26n.pt \
  OUTPUT=runs/players/match-pose-gpu
```

Le chargement CUDA importe d'abord Torch pour charger les bibliothèques CUDA 13
du même environnement. Le provider CUDA est contrôlé ; une initialisation
échouée ou une erreur d'exécution arrête le run au lieu de basculer silencieusement
toute l'inférence sur CPU. Des opérations du graphe peuvent utiliser le provider
CPU secondaire. La pose reste en FP32, TF32 désactivé pour limiter les écarts
numériques entre lots. La CLI expose `--pose`, `--no-pose`, `--pose-weights`
et `--pose-device`.

## Réglages

```yaml
pose:
  enabled: true
  weights: models/players/rtmpose-m.onnx
  device: cpu
  cpu_threads: 4
  batch_size: 16
  min_detection_confidence: 0.25
  keypoint_min_confidence: 0.3
  min_bbox_area: 256.0
  max_players: 32
```

Une frame est traitée à la fois. Les boîtes de score suffisant et d'aire minimale
en pixels source sont candidates. Si elles dépassent `max_players`, les plus
grandes sont retenues, puis traitées par lots bornés. Les résultats retrouvent
toujours l'ordre initial des détections. `score_floor` doit être inférieur ou
égal à `min_detection_confidence`. Aucune inférence de pose n'est lancée pour
une frame sans candidat. Le seuil d'affichage vidéo `score_threshold` reste
indépendant du filtrage de pose et des données exportées.

## Sorties et lecture des points

| Fichier | Contenu |
| --- | --- |
| `poses.jsonl` | Une ligne par frame, détections, IDs et poses, y compris les frames vides |
| `pose.json` | Modèle, empreinte, provider, géométrie, noms des points, connexions et paramètres |
| `annotated.mp4` | Boîtes, IDs et squelettes partageant la couleur de chaque track |
| `result.json` | Nombre de poses calculées, points valides et empreintes des artefacts |
| `predictions.jsonl`, `tracks.jsonl` | Détections et suivi conservant leurs contrats précédents |

Une ligne `poses.jsonl` porte `schema_version: 1`,
`artifact_type: players_video_poses`, `source_sha256`, `pose_run_id`,
`tracking_run_id`, `segment_id`, `frame_index`, `timestamp_seconds` et
`detections`. Chaque détection conserve sa boîte XYXY, classe, confiance et ID.
`detection_index` désigne sa position dans la frame, même sans ID confirmé.
L'identité temporelle reste `(tracking_run_id, track_id)` ; la pose ne corrige
pas les erreurs d'association du tracker.

Le champ `pose` contient `format: coco17`, `coordinate_space: source`,
`keypoints` (17 couples `[x, y]` en pixels), `scores` (17 scores bruts),
`valid` (17 booléens), `valid_keypoints` et `score` (moyenne des scores).
Ces scores sont la moyenne des maxima SimCC X/Y, **pas des probabilités calibrées**.
Un point est valide si son score positif atteint le seuil et s'il est dans
l'image. Les positions hors image restent exportées mais invalides ; un score
non positif donne une position `null`. Seuls les points valides et les segments
dont les deux extrémités sont valides sont dessinés.

| Index | Point | Index | Point |
| ---: | --- | ---: | --- |
| 0 | Nez | 9 | Poignet gauche |
| 1 | Œil gauche | 10 | Poignet droit |
| 2 | Œil droit | 11 | Hanche gauche |
| 3 | Oreille gauche | 12 | Hanche droite |
| 4 | Oreille droite | 13 | Genou gauche |
| 5 | Épaule gauche | 14 | Genou droit |
| 6 | Épaule droite | 15 | Cheville gauche |
| 7 | Coude gauche | 16 | Cheville droite |
| 8 | Coude droit | | |

Gauche/droite désigne le corps de la personne. `pose_status` explique le résultat :
`estimated`, `low_keypoint_confidence` (pose calculée sans point valide),
`low_detection_confidence`, `small_bbox` ou `player_limit` (ces trois derniers
donnent `pose: null`). `posed_observations` compte les poses calculées, même si
aucun point n'est valide. Une personne non confirmée par le tracker peut avoir
une pose avec `track_id: null`.

Une interruption conserve `poses.partial.jsonl` et les autres sorties partielles.
La relance exige un nouveau dossier ; la reprise au milieu de la vidéo n'est pas
implémentée. Il n'y a pas de lissage temporel, de squelette 3D, de mains détaillées
ni de réidentification. Les points occultés restent des estimations, même avec
un score élevé. Un détecteur COCO peut proposer des arbitres et des spectateurs,
qui recevront aussi un squelette si leurs boîtes passent les filtres.

## Modèle et validation

Modèle officiel :
[RTMPose-M body7 256×192 ONNX](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip).
SHA-256 du fichier ONNX :
`5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c`.
Références : [RTMPose/MMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose),
[rtmlib](https://github.com/Tau-J/rtmlib),
[provider CUDA ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
Le recadrage ajoute une marge de 1.25, conserve le ratio 192:256, utilise BGR
et décode SimCC avec un facteur 2.

```bash
make test-players-pose PLAYERS_PYTHON=.venv-players-cpu/bin/python
# Test réel local supplémentaire ; aucun téléchargement pendant les tests :
PLAYERS_POSE_WEIGHTS=models/players/rtmpose-m.onnx \
  make test-players-pose PLAYERS_PYTHON=.venv-players-cpu/bin/python
# Même vérification sur GPU :
PLAYERS_POSE_WEIGHTS=models/players/rtmpose-m.onnx PLAYERS_POSE_DEVICE=cuda:0 \
  make test-players-pose PLAYERS_PYTHON=.venv-players/bin/python
```

Validation locale du 26 septembre 2026 : 14 tests de contrats et une intégration
avec le vrai ONNX réussis, sur CPU puis CUDA. La suite joueurs complète découvre
133 tests : 123 réussis, 10 intégrations optionnelles désactivées. Les tests de
pose couvrent géométrie, filtres, lots, ordre/IDs, points invalides, installation,
refus du fallback CUDA, export vidéo et interruption. Ils sont découverts par
la CI existante ; l'intégration avec poids reste explicitement activable.

Sur les neuf personnes de la première frame HOU/LAL, comparaison avec le chemin
NBA/rtmlib CPU : crops normalisés, coordonnées et scores **identiques**.
Deux essais complets sur les 120 premières frames, en CPU et sur RTX 5080,
produisent chacun 1 101 poses et 18 336 points valides. Les boîtes, scores, ordre
et IDs sont conservés entre les sidecars de suivi et de pose. Une frame annotée
a été inspectée visuellement. Ces essais valident le fonctionnement technique,
sans mesurer la précision des points contre des annotations de référence.

Artefacts locaux : `runs/players/pose-validation/`, notamment
`nba-parity.json`, `hou-lal-cpu-final/` et `hou-lal-gpu/` avec `preview.jpg`.
Le détecteur d'essai utilise les poids locaux `/home/erwan/NBA/models/yolo26n.pt` ;
le modèle de pose est installé dans Vyneros. Le test CUDA termine avec le
provider CUDA actif ; cuDNN émet des diagnostics « No execution plans support
the graph » pendant la recherche de plans, sans empêcher ce run de se terminer.
Les débits enregistrés incluent chargement et encodage, sans préchauffage ;
ces clips courts ne constituent pas un benchmark stable.
