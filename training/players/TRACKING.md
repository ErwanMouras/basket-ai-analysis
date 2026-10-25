# Suivi des joueurs et track_id

L’inférence vidéo peut maintenant suivre les boîtes de YOLO26 **ou RF-DETR** avec
**BoT-SORT sans ReID**, ou **ByteTrack**. Les IDs sont écrits dans `tracks.jsonl`
et affichés sur `annotated.mp4`. Aucun entraînement du tracker n’est nécessaire.

Les [squelettes RTMPose-M](POSE.md) s'ajoutent avec `make pose-players` après
installation du modèle. `poses.jsonl` conserve les boîtes et IDs de `tracks.jsonl`
et ajoute les 17 points COCO, leurs scores et leur validité.

La [calibration NBA](../court/README.md) s'ajoute avec `make setup-court`, puis
`make calibrate-court VIDEO=... CHECKPOINT=... OUTPUT=...`. Elle produit les
positions en mètres, les lignes du terrain et une vue de dessus. Avec cette
option, les coupes détectées réinitialisent également les tracks.

## Lancer le suivi

Dans l’environnement joueurs décrit dans [TRAINING.md](TRAINING.md), installer
le lock actualisé incluant `lap==0.5.12`. Pour un environnement déjà installé avec
les autres versions du lock :

```bash
uv pip install --python .venv-players/bin/python lap==0.5.12
make track-players \
  VIDEO=/videos/match.mp4 \
  CHECKPOINT=/poids/yolo26n.pt \
  OUTPUT=runs/players/match-botsort
```

Le répertoire de sortie doit être nouveau. `MAX_FRAMES=300` limite un essai.
`TRACKER=bytetrack` choisit l’autre tracker. Pour CPU uniquement, passer
`PLAYERS_PYTHON=.venv-players-cpu/bin/python` après installation du lock CPU.
Le détecteur de la recette par défaut est YOLO26 Nano, CPU, résolution 640.

Pour un autre modèle, copier [configs/predict_video.yaml](configs/predict_video.yaml)
et adapter `model`, `variant`, `weights`, `source_class`, `device` et `resolution`.
Par exemple RF-DETR Nano : `model: rfdetr`, `variant: rfdetr_nano`. Le tracker
consomme les mêmes boîtes source pour les deux familles, sans réexécuter le détecteur.

```bash
make track-players CONFIG=/recettes/ma-video.yaml OUTPUT=runs/players/autre-run
# Équivalent Python :
.venv-players/bin/python -m training.players.predict \
  --config /recettes/ma-video.yaml --tracker botsort \
  --output runs/players/autre-run --max-frames 300
```

`make predict-players ... TRACKER=none` désactive le suivi. Les anciennes recettes
sans bloc `tracking` restent en détection seule ; la recette vidéo fournie active
désormais BoT-SORT. `track-players` active explicitement le tracker demandé,
même si la recette a `enabled: false`.

## Sorties et sens d’un ID

| Fichier | Contenu |
| --- | --- |
| `predictions.jsonl` | Détections brutes, contrat existant inchangé |
| `tracks.jsonl` | Une ligne par frame, mêmes boîtes/scores/classe, avec `track_id` entier positif ou `null` |
| `tracking.json` | Tracker, versions, réglages, buffer effectif, empreinte de l’adaptateur et provenance du détecteur |
| `annotated.mp4` | Boîtes avec couleur stable et label `#ID`, ou `player ?` pour une observation non confirmée |
| `result.json` | Nombre d’observations associées, débit global et empreintes des sorties |

Exemple de ligne de `tracks.jsonl` (les valeurs SHA-256/run ci-dessous sont abrégées) :

```json
{
  "schema_version": 1,
  "artifact_type": "players_video_tracks",
  "coordinate_space": "source",
  "source_sha256": "...",
  "tracking_run_id": "...",
  "segment_id": 0,
  "frame_index": 12,
  "timestamp_seconds": 0.4,
  "detections": [
    {"class_id": 0, "bbox": [10.0, 20.0, 110.0, 220.0], "confidence": 0.91, "track_id": 7}
  ]
}
```

Une identité de track est **`(tracking_run_id, track_id)`**. Un nouveau run
recommence ses propres IDs ; les IDs de deux runs ne sont pas comparables.
Dans un même run, une remise à zéro ne réutilise pas les IDs antérieurs.

Un ID signifie une continuité estimée de boîtes, **pas un numéro de maillot ou
un joueur reconnu sur tout le match**. Les nouvelles tracks après la première
frame demandent confirmation sur une observation suivante ; les scores trop
faibles ou les tracks non confirmées gardent `null`. Les trous entre entiers sont
normaux. Les frames vides sont écrites et transmises au tracker pour faire vieillir
ses tracks. Une occultation brève peut conserver un ID ; une disparition prolongée
en crée généralement un nouveau. Aucune boîte prédite sans détection courante
n’est exportée comme observation réelle.

Les boxes exportées restent celles du détecteur, avec un ID associé ; ce ne sont
pas les boîtes lissées du filtre de Kalman. Le seuil d’affichage ne retire pas
les observations du JSONL. Si le détecteur fournit des arbitres ou du public,
le tracker peut aussi les suivre : il ne corrige pas la classe du détecteur.

## Réglages

```yaml
tracking:
  enabled: true
  tracker: botsort
  track_high_thresh: 0.25
  track_low_thresh: 0.1
  new_track_thresh: 0.25
  track_buffer_seconds: 1.0
  match_thresh: 0.8
  fuse_score: false
  gmc_method: sparseOptFlow
  reset_frames: []
```

Le tracker utilise une association sur détections fortes, puis tente de récupérer
les tracks présentes avec les faibles scores. `score_floor` doit rester inférieur
ou égal à `track_low_thresh` ; un seuil de détection trop haut supprimerait les
candidats nécessaires à la seconde association. `new_track_thresh` doit être
au moins égal à `track_high_thresh`.

`fuse_score: false` est volontaire : dans Ultralytics **8.4.39**, la seconde
association fusionne aussi les scores lorsque cette option est active, avec une
distance maximale fixe de 0.5. Cela empêche notamment une observation à 0.2 de
récupérer sa track, même si les boîtes coïncident. La fixture de faible confiance
vérifie la récupération avec le réglage livré.

Le buffer vaut `ceil(fps × track_buffer_seconds)` frames source. Les tracks
perdues expirées sont retirées avant association. `gmc_method: sparseOptFlow`
compense les mouvements de caméra dans BoT-SORT ; `none` le désactive. ByteTrack
n’utilise pas cette compensation. Il n’y a ni modèle ReID ni téléchargement
de poids de tracking. Le coût du tracker est CPU, indépendamment du device du
détecteur.

`reset_frames: [300, 720]` vide l’état **avant** ces frames (indices à partir de
zéro) et incrémente `segment_id`. À utiliser pour des changements de plan connus.
Sans l'option `court`, les coupes doivent être déclarées explicitement. Avec
`court.enabled: true` et `court.auto_cuts: true`, une détection heuristique des
coupes est active, avec protection contre les flashs photo. Elle ne garantit
pas de repérer tous les montages ou fondus ; les limites connues peuvent toujours
être déclarées dans `reset_frames`. Les replays et la réidentification après une
longue sortie restent à traiter avant d'interpréter les IDs comme une identité
persistante. L'OCR optionnel possède sa propre mémoire par segment.

Les images sont traitées une à une ; les fichiers sont écrits immédiatement.
Le tracker ne conserve que ses états actifs/perdus et l’historique de suppression
borné par le backend. Les sorties interrompues restent
`tracks.partial.jsonl`, `predictions.partial.jsonl`, `annotated.partial.mp4`.
La reprise au milieu d’une vidéo n’est pas prise en charge : recommencer dans
un nouveau dossier pour reconstruire exactement l’état du tracker.

## Choix technique après revue des trackers actuels

Revue du 26 septembre 2026. Le tableau public BoxMOT fournit notamment ces HOTA
sur SportsMOT val :

| Tracker | HOTA annoncé | Position dans cette intégration |
| --- | ---: | --- |
| OccluBoost | 83.17 | Candidat ultérieur à mesurer localement |
| HybridSORT | 81.14 | Candidat ultérieur à mesurer localement |
| Deep OC-SORT | 79.51 | Candidat avec apparence/ReID à évaluer |
| BoT-SORT | 76.93 | Famille retenue pour caméra mobile |
| BoostTrack | 76.32 | Autre candidat de comparaison |
| ByteTrack | 67.93 | Alternative simple et légère livrée |

Source : [benchmark officiel BoxMOT](https://github.com/mikel-brostrom/boxmot#benchmark-results).
Ces scores concernent **ses implémentations et son protocole**, pas notre
adaptateur Ultralytics sans ReID, et ne prédisent pas les résultats sur HOU/LAL
ou FFBB. Ils montrent aussi que BoT-SORT ne doit pas être présenté comme le
meilleur tracker actuel dans l’absolu.

Ultralytics propose désormais également OC-SORT, Deep OC-SORT, FastTracker et
TrackTrack à partir de 8.4.63. Le dépôt est épinglé à **8.4.39** : cette livraison
utilise BoT-SORT/ByteTrack déjà présents pour conserver l’environnement des
détecteurs validés. Source : [documentation officielle du suivi](https://docs.ultralytics.com/modes/track).

BoT-SORT combine association de mouvement et compensation caméra, avec apparence
optionnelle ; notre profil désactive cette dernière. Référence primaire :
[article BoT-SORT](https://arxiv.org/abs/2206.14651).
Les versions Ultralytics/LAP sont contrôlées avant import pour éviter une
installation automatique de LAP pendant l’inférence. Un changement de version
nécessite de rejouer les tests de l’adaptateur.

Deux corrections d’intégration sont testées : les indices de détection sont
préservés à travers les filtres de confiance du backend, et le compteur d’IDs
est propre à chaque instance plutôt que global au processus. Il n’y a aucun
import du code NBA.

## Validation réalisée

```bash
make test-players-tracking PLAYERS_PYTHON=.venv-players-cpu/bin/python
```

**14 tests dédiés réussis** : deux trackers réels, ordre des boîtes, filtrage et
faibles scores, occultation/reprise, expiration, nouvelle arrivée, coupe explicite,
instances intercalées, validation des entrées, vidéo et interruption. Les doubles
de détecteurs YOLO/RF-DETR vérifient le contrat commun indépendamment des poids.
Les tests du tracker nécessitent le lock joueurs ; ils sont explicitement
désactivés si cet environnement est absent.

Suite joueurs complète hors réseau : **109 réussis, 9 intégrations optionnelles
désactivées** (118 tests découverts). Suite balle : **56 réussis, 13 optionnels
désactivés**. Les environnements CPU et CUDA passent `uv pip check` après ajout
de LAP. Aucun entraînement ni essai GPU n’a été lancé pour cette livraison.

Un smoke du **vrai adaptateur RF-DETR Nano**, avec les poids aléatoires locaux
de validation, a aussi traité quatre frames à résolution 128 avec BoT-SORT et
réseau bloqué. Il vérifie le raccordement et produit les sidecars ; aucune
observation n’a reçu d’ID avec ces poids aléatoires. Ce smoke ne mesure aucune
qualité de détection ou d’association.

Deux essais réels, séquentiels, sur les **120 premières frames HOU/LAL**, source
1920×1080 à 30 FPS, YOLO26 Nano local, CPU, FP32, entrée 640, 4 threads :

| Profil | Observations avec ID | Temps complet | Débit complet |
| --- | ---: | ---: | ---: |
| BoT-SORT sans ReID | 1 255 | 6.99 s | 17.18 images/s |
| ByteTrack | 1 243 | 5.84 s | 20.56 images/s |

Les `predictions.jsonl` des deux essais ont le même SHA-256 : seuls suivi et
affichage diffèrent. Le temps inclut chargement, décodage, détection, tracking,
dessin, encodage et écritures. Ces clips courts incluent le démarrage et ne
constituent pas un benchmark stable de vitesse ni une mesure HOTA/IDF1.
Le nombre d’observations associées n’est pas une métrique de qualité d’identité.

Artefacts locaux : `runs/players/tracking-validation/`, avec `tests.json`,
`hou-lal-botsort/` et `hou-lal-bytetrack/`. Les poids de cet essai ont été lus dans
`/home/erwan/NBA/models/yolo26n.pt` et leur empreinte est dans `model.json` ; c’est
une ressource locale d’essai, aucune dépendance de code ou chemin NBA n’est
nécessaire pour utiliser le suivi avec ses propres poids. La vidéo BoT-SORT a
été inspectée sur une frame ; la continuité des identités du clip entier reste
à évaluer contre une vérité terrain de tracking.
