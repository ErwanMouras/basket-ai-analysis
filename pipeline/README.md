# Analyse vidéo NBA

Entrée unique vidéo, effectif facultatif, six capacités de perception et sortie
`statistics.json` contenant la distance observable en mètres. Aucun entraînement,
accès à MLflow, accès DVC ou rendu vidéo n'est requis à l'exécution.

## Utilisation

Depuis la racine du dépôt, utiliser l'environnement joueurs existant :

```bash
# Une seule fois, installation explicite du checkpoint provisoire joueurs/arbitres.
.venv-players/bin/python -m pipeline.setup

# Vidéo seule : suivi anonyme et OCR sans effectif.
.venv-players/bin/python -m pipeline analyze \
  --video datas/train/HOU_LAL/G1/video.mp4 \
  --config configs/analysis.yaml \
  --output runs/analysis/my-video

# Effectif explicitement fourni, même format que training/jersey/README.md.
.venv-players/bin/python -m pipeline analyze \
  --video datas/train/HOU_LAL/G1/video.mp4 \
  --roster training/jersey/examples/hou_lal_g1_2026/jersey.json \
  --config configs/analysis.yaml \
  --output runs/analysis/my-video-roster \
  --max-frames 150

# Recalcul sans charger les modèles ; vérifie l'intégrité des observations.
.venv-players/bin/python -m pipeline recompute \
  --run runs/analysis/my-video --output runs/analysis/recomputed.json
```

Le dossier d'analyse doit être nouveau. Une erreur ou interruption laisse les
artefacts `.partial.*` et les diagnostics dans `run.json` / `progress.json`.
Le consommateur vérifie le statut final de `run.json` avant de lire les sorties.
La publication est atomique par fichier, le manifeste final sert de validation
de l'ensemble. Pas de reprise de tracking en milieu de vidéo.

## Visualisation

Lancer `make view-analysis`, puis ouvrir <http://127.0.0.1:8765> pour explorer
les runs terminés : vidéo avec calques de détection, terrain projeté et
statistiques progressives. Voir le [guide du visualiseur](../viewer/README.md)
pour les dossiers personnalisés et les limites des métriques.

Le profil [analysis.yaml](../configs/analysis.yaml) utilise les poids déjà présents
pour terrain, RTMPose et PARSeq. La balle référence un checkpoint V5 local dans
`runs/ball/` : remplacer ce chemin par le checkpoint choisi si nécessaire. Elle
requiert aussi la référence épinglée `.external/tracknet_v5`. Les commandes
existantes `make setup-court`, `make setup-players-pose`, `make setup-jersey`
installent explicitement les trois autres modèles. L'inférence ne télécharge rien.
Le profil GPU utilise l'environnement CUDA joueurs ; un profil CPU doit mettre
chaque `device` à `cpu` et employer l'environnement CPU joueurs.

## Configuration et modèles

- `people.adapter`: `yolo` (multiclasse) ou `rfdetr` (une classe source explicite,
  variantes et poids compatibles avec le parcours joueurs existant).
- `people.class_roles` associe des IDs de classes du modèle à `player`, `referee`
  ou `unknown`. Un modèle COCO `person` ne peut pas être présenté comme joueur.
  Le score de détection multiclasse est celui de la classe prédite ; aucune
  probabilité de rôle indépendante n'est inventée.
- `ball.adapter: tracknet_v5` accepte V5 et V5+TOTNet enregistrés par les trainers
  locaux. Le dataset d'entraînement/évaluation n'est pas requis à l'inférence.
- `required` impose les capacités nécessaires avant le run et arrête le traitement
  en cas de panne de l'une d'elles. Le profil fourni exige les six capacités.
  Retirer une capacité de cette liste permet sa dégradation explicite ; utiliser
  `enabled: false` pour désactiver balle, terrain, pose ou OCR intentionnellement.
- `include_intervals` et `exclude_intervals` contiennent des plages `[début, fin]`
  en secondes vidéo, début inclus et fin exclue. Chaque frontière coupe les
  trajectoires, même lorsqu'elle tombe entre deux images.
- `tracking.reset_frames` indique des coupes connues, indices à partir de zéro.
  Les coupes automatiques sont fournies par le module terrain quand il est actif.
- `timestamp_policy: source` lit les timestamps OpenCV du flux et refuse une
  progression invalide. `fps` est un choix explicite pour les vidéos connues à
  cadence constante. Le buffer du tracker reste exprimé en FPS nominaux ; la
  calibration, les votes OCR et la distance utilisent les timestamps vidéo.

### Baseline joueurs/arbitres provisoire

Checkpoint **E-BARD YOLOv8n**, auteur Gabriele Giudici, fiche modèle indiquant
CC-BY-4.0 ; provenance et limites :
[fiche originale](https://huggingface.co/GabrieleGiudici/E-BARD-detection-models).
Révision `3f4789c4431aa73269f60107a4ba0a5f86b7af8b`, fichier
`BODD_yolov8n_0001.pt`. SHA-256 vérifié à l'installation :
`dfe3534d51bb21024d1a400c37f0c1fbf0c8b96ea9a56a5f3cb5454813bfd641`.
Le checkpoint expose `0=basketball`, `1=hoop`, `2=player`, `3=referee` ; seuls les
deux derniers sont utilisés. La balle conserve le modèle temporel du POC.
Cette intégration ne constitue pas une sélection définitive ni un benchmark de
précision sur nos matchs. Le modèle reste remplaçable sans modifier les statistiques.

## Graphe et ressources

Décodage unique avec prélecture bornée. Inférences personnes et repères terrain
indépendantes ; la pose part dès les boîtes disponibles pendant la calibration
et le tracking CPU, puis est jointe par indice de détection. La balle travaille
en parallèle de la pose/OCR lorsque le budget de device le permet. Le tracking
utilise une association séparée par rôle, puis des IDs globaux pour le run.

`workers`, `prefetch`, `cpu_threads` et `gpu_concurrency` bornent le travail.
Chaque instance de modèle est protégée contre les appels concurrents ; les
modèles ne sont chargés qu'une fois. Les files ne jettent pas de frames en cas
de saturation. Les résultats sont associés à leur frame source et écrits dans
l'ordre. Une seule barrière par frame finalise son observation ; elle est
nécessaire au journal unifié de cette V1, et limite le recouvrement de certains
travaux. Aucun rendu/encodage vidéo n'est exécuté.

TrackNet tourne dans un processus persistant créé avec `spawn`, isolant ses
imports globaux et sa configuration Torch. Il reçoit des triplets redimensionnés.
Politique **causale** : trois images passées du même segment, dernière sortie
de la fenêtre pour la frame courante. Les deux premières frames de chaque
segment portent `insufficient_segment_context`. Pas de mélange entre segments,
pas de contexte futur ni moyenne de fenêtres comme dans l'évaluateur hors ligne.

L'OCR utilise un budget en temps vidéo pour rendre les crops sélectionnés
comparables entre séquentiel et parallèle. `max_duty_cycle` n'impose donc pas de
limite de temps réel dans ce pipeline ; les anciens parcours conservent leur
comportement. Les gardes mémoire du lecteur restent actives et leurs reports
figurent dans `jersey_reads.jsonl`. Les threads Torch sont fixés au lancement,
pas modifiés au milieu d'un travail concurrent.

`--mode sequential` fournit le témoin. `run.json` sépare chargement, traitement,
finalisation, appels/attentes par modèle et RSS maximal. Les pics CUDA mesurés
sont ceux de PyTorch (processus principal et worker balle) : ils excluent ONNX
Runtime et les contextes CUDA. Ils ne représentent pas la VRAM totale utilisée.

## Contrats de sortie

Schémas JSON versionnés : [observation](schemas/observation.schema.json) et
[statistiques](schemas/statistics.schema.json). Les invariants temporels et
référentiels sont aussi vérifiés par `contracts.py` et le moteur de distance.

| Artefact | Contenu |
| --- | --- |
| `statistics.json` | Sujets, identité éventuelle, références des tracks, distance, durées, couverture, exclusions et limites |
| `observations.jsonl` | Une ligne par frame : personnes, rôle, boîtes, pose, maillot, positions métriques, balle, calibration, statuts |
| `tracks.jsonl` | Une ligne par contribution : track/sujet, temps, positions brute/filtrée, distance de l'intervalle ou motif de rupture |
| `jersey_reads.jsonl` | Crops retenus, frame d'origine, lecture, confiance et décision OCR |
| `team_votes.json` | Votes numériques et couleurs par track, avec images et totaux permettant d'auditer la majorité |
| `roster.json` | Copie figée, uniquement si un effectif est fourni |
| `run.json` | Source/empreinte, configuration, modèles/empreintes, capacités, qualité d'exécution, performances, hashes des sorties |
| `progress.json` | État et nombre de frames terminées |

Les coordonnées des boîtes, squelettes et balle sont en pixels de la vidéo
source. Les joueurs sont projetés en mètres, avec convention NBA issue du module
terrain. Le repère est propre au segment caméra. La balle n'est pas projetée au
sol par homographie. Les observations peuvent porter `null` et un motif explicite.
`run.json` contient `display_colors` (équipes, groupes anonymes, arbitres et
inconnus) ; chaque personne dans `observations.jsonl` porte `display_color`.
La couleur d'une équipe provient du champ facultatif `color` du roster, ou de
la palette stable du pipeline. Elle ne préjuge pas d'une équipe quand seul un
groupe de maillots anonyme est connu.

Après les votes de numéro et de couleur, une continuité spatiale peut attribuer
une nouvelle portion de track à un joueur déjà identifié, ou retrouver vers
l'arrière une portion anonyme juste avant une track identifiée. Le découpage suit les
séquences de couleur cohérente : un changement durable de maillot
coupe la track, même lorsque le tracker réutilise son identifiant. L'OCR analyse
une image sur deux jusqu'à confirmation du numéro dans chaque séquence, tandis
que la couleur du torse reste analysée sur chaque image. Les deux portions
doivent être dans le même segment caméra, séparées d'au plus 0,2 s et proches
dans **l'image vidéo**. Les coordonnées issues de l'homographie restent dans
la preuve, sans intervenir dans la décision. Un joueur proche de la même équipe
ou d'équipe inconnue bloque la liaison ; un adversaire de couleur confirmée
n'est pas confondu avec le joueur suivi. La couleur est évaluée sur la portion
de track concernée, car un ID de tracker peut être réutilisé pour l'autre équipe.
Une contradiction de numéro ou de couleur bloque la liaison ; le retour d'une
track déjà identifiée arrête la liaison avant la frame de chevauchement. Les
liaisons inverses exigent aussi la disparition effective de l'ancienne track
au point de raccord. Plusieurs candidats proches rendent la transition ambiguë
et empêchent l'attribution. Les images sans couleur lisible dans une portion continue et propre conservent
l'identité directe. `spatial_handoff` dans l'observation et `spatial_handoffs`
dans `run.json` documentent chaque liaison.

Les champs issus des adaptateurs pose/OCR sont conservés sans perdre leurs
confiances ni statuts propres. `ok`, `not_detected`, `disabled`, `unavailable` et
`error` distinguent disponibilité de la capacité et contenu observé. Le schéma
réserve aussi `not_scheduled` pour les futures politiques de sous-échantillonnage.

## Signification de la distance

La métrique additionne les déplacements sur le parquet, après filtre exponentiel
temporel et seuil spatial anti-bruit. Valeurs initiales configurables, **non
calibrées par une vérité terrain de distances** : constante de temps 0,12 s,
seuil spatial 0,025 m, vitesse maximale de contrôle 12 m/s, intervalle maximal
0,2 s. Un filtre peut sous-estimer les virages ; les erreurs de localisation
peuvent aussi surestimer la distance. Les seuils ne garantissent pas sa précision.

Coupe caméra, trou de détection/projection, frontière d'inclusion, changement de
point au sol, conflit d'identité ou vitesse aberrante rompent la trajectoire.
Aucune distance n'est ajoutée à travers ces ruptures. Une absence de mesure vaut
`null`, jamais zéro. Zéro exige au moins une paire de positions valide.

`measured_duration_s` additionne les durées des paires valides.
`tracked_duration_s` additionne les durées entre observations consécutives du
même sujet dans les plages incluses, même quand la projection est indisponible.
Les absences de détection ne sont pas incluses dans ce dénominateur ; leur retour
est signalé par `tracking_gap`. `coverage_ratio` est le rapport des deux durées,
pas une estimation du pourcentage de présence réelle du joueur sur le terrain.

Sans roster : une identité anonyme par track et segment. Avec roster : les votes
sur tout le run et les liaisons spatiales prudentes attribuent les portions de
track et recalculent les statistiques par joueur. Les contributions d'une même
identité peuvent être additionnées entre segments ; leurs coordonnées ne sont
jamais reliées. Numéros partagés sans équipe, transitions ambiguës et conflits
de tracks simultanées restent anonymes. Les IDs erronés du tracker ne peuvent
pas tous être détectés sans réidentification.

Le statut du run décrit son exécution ; chaque sujet a son propre
`measurement_status`. Un run `completed` peut donc contenir des distances
`partial` ou `unavailable`. Les replays, ralentis et sorties de champ empêchent
d'assimiler cette métrique à une distance totale de match.

## Vérification

```bash
.venv-players/bin/python -m unittest discover -s pipeline/tests -v
make test-court PYTHON=.venv-players/bin/python
make test-jersey PYTHON=.venv-players/bin/python
make test-players PYTHON=.venv-players/bin/python
make test-ball PYTHON=.venv-players/bin/python
```

Les tests analytiques couvrent distance, bruit, virages, ruptures, conflits,
exclusions, rôle arbitre et couverture. Les tests d'intégration utilisent une
vraie vidéo synthétique et des adaptateurs injectés, sans poids ni GPU.
Le [rapport de validation](VALIDATION.md) décrit séparément les essais réels.
