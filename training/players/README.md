# Détection des joueurs

Les phases 1 à 7 sont implémentées : contrats de données, infrastructure commune,
annotation de boîtes, préannotation, exports reproductibles YOLO/COCO et
entraînement YOLO26/RF-DETR avec MLflow, évaluation COCO commune et comparaison,
orchestration locale, inférence vidéo et registre MLflow, validation finale et CI.
Le [guide de validation](VALIDATION.md) fournit l'installation CPU/CUDA, une démo
sans poids externes, les commandes de CI, MLflow et la restauration DVC.
Le [bilan mesuré](VALIDATION_RESULTS.md) distingue les tests techniques de la
qualité sur les matchs et donne le volume réellement annoté.
Le [guide du parcours local](PIPELINE.md) décrit `pipeline-players`,
`predict-players`, `register-players`, la progression et la reprise.
Le [guide de tracking](TRACKING.md) décrit `track-players` : BoT-SORT ou ByteTrack,
IDs dans `tracks.jsonl` et sur la vidéo, compatibles avec les deux détecteurs.
Le [guide des squelettes](POSE.md) décrit `pose-players` : RTMPose-M, 17 points
COCO par personne, associés aux mêmes IDs, exportés et dessinés sur la vidéo.
Le [guide des numéros de maillot](../jersey/README.md) décrit `predict-jersey` :
lecture PARSeq sur crops sélectionnés, confirmation temporelle et budgets CPU/GPU.
Le [guide de calibration NBA](../court/README.md) décrit `calibrate-court` :
homographies avec caméra mobile, dimensions réglementaires, positions en mètres,
superposition des lignes et vidéo du terrain vu du dessus.
Le [guide d'évaluation](EVALUATION.md) décrit les métriques, le protocole de vitesse,
les rapports, les poids préentraînés et le gel du test final.
Le [guide d'entraînement](TRAINING.md)
décrit l'environnement séparé, les recettes, le chargement de poids sans
entraînement, les checkpoints et la reprise stricte. Le
[plan](../../docs/PLAN_PLAYERS.md) décrit les sept phases et leurs limites.

## Annoter une vidéo

Depuis la racine du dépôt :

```bash
make annotate-players MEDIA=datas/train/HOU_LAL/G1/video.mp4
```

`VIDEO=...` reste accepté comme alias de `MEDIA`. L'éditeur ouvre une fenêtre Tk
locale, avec les dépendances déjà utilisées par l'annotateur balle (OpenCV,
Pillow, Tk). Il nécessite un affichage graphique ; les tests de données n'en ont
pas besoin. Les écritures concurrentes utilisent les verrous Linux `flock`.

Par défaut, il sélectionne une frame toutes les 30 frames et ajoute toutes les
frames qui possèdent déjà des annotations. Pour travailler sur toutes les frames :

```bash
make annotate-players MEDIA=datas/train/HOU_LAL/G1/video.mp4 PLAYER_STEP=1
```

Pour reprendre les 200 premières frames de POR_SAS :

```bash
make annotate-players MEDIA=datas/val/POR_SAS/G5/video.mp4 PLAYER_STEP=1 PLAYER_STOP=200
```

`PLAYER_START` est inclusif, `PLAYER_STOP` exclusif et facultatif. Tous les indices
commencent à zéro. La sélection ne valide pas les frames intermédiaires.

Le média doit se trouver sous `PLAYERS_ROOT` (défaut : `datas`). Son `match_id`
vient du sidecar existant, de `.meta.yaml`, de `match.yaml` ou de `MATCH_ID=...`.
`VENUE_ID` et `PLAYER_SPLIT` permettent également de fournir les identités ; une
contradiction avec les métadonnées ou le dossier `train/val/test` est refusée.
Le SHA-256 et les dimensions sont vérifiés à l'ouverture. Les chemins du sidecar
sont relatifs à `PLAYERS_ROOT`, qui peut être déplacé avec tout son contenu.

Le résultat est enregistré à côté du média : `video.mp4.playersann.json`.

### Commandes de l'éditeur

| Action | Geste / touche |
| --- | --- |
| Créer une boîte | Cliquer-glisser dans une zone libre |
| Créer une boîte superposée | Activer « Nouvelle boîte » (`N`), puis glisser |
| Sélectionner / déplacer | Cliquer / glisser dans une boîte ; la plus petite est prioritaire |
| Redimensionner | Glisser un coin blanc de la boîte sélectionnée |
| Choisir joueur / arbitre | `J` / `A` : reclasse la boîte sélectionnée ou choisit la classe des nouvelles boîtes |
| Supprimer | `Suppr` ou `Retour arrière` |
| Occultation / troncature | `O` / `T` : inconnue → oui → non |
| Vérifier toute la frame | `V`, après correction de tous les joueurs et arbitres |
| Réouvrir la vérification | `R` |
| Naviguer | Flèches gauche/droite, curseur ou boutons |
| Prochaine frame à vérifier | `F`, dans la sélection du média courant |
| Zoomer | Boutons zoom ; barres de défilement pour déplacer la vue |
| Annuler le geste en cours | `Échap` |

La sauvegarde intervient après chaque modification terminée. Une frame modifiée
repasse `in_progress`, même si elle était validée. Valider une frame sans boîte
déclare explicitement qu'elle ne contient aucun joueur ni arbitre. Une frame absente,
non annotée ou préannotée n'est jamais un négatif d'entraînement.

Une erreur d'écriture conserve le fichier précédent et l'état en mémoire. Si un
autre éditeur ou outil a changé le sidecar, le prochain enregistrement est refusé :
fermer et rouvrir le média pour récupérer le nouvel état. Les fichiers `.lock`
restent locaux, ignorés par Git et DVC.

### Convention d'annotation

Annoter tous les joueurs participant au jeu et les arbitres visibles, avec une boîte corporelle alignée
aux axes de l'image, ajustée à leur étendue visible. Inclure un joueur encore
identifiable lorsqu'il est partiellement masqué ; renseigner `occluded` et ne pas
inventer les limites des parties invisibles. Une personne entièrement invisible
n'a pas de boîte. `truncated` indique une coupure par le bord de l'image.

Les joueurs qui sortent momentanément du terrain pendant l'action restent cibles.
Un arbitre est annoté en classe `1` (`referee`), un joueur en classe `0` (`player`).
Entraîneurs, public et remplaçants sur le banc ne sont pas annotés. Si la
situation est ambiguë, laisser la frame en cours au lieu de la valider incomplète.
Pas de numéro de maillot, équipe ou identifiant de tracking dans cette phase.

## Préannoter avec un checkpoint

Pour proposer les joueurs **et les arbitres** avec le checkpoint E-BARD local,
puis ouvrir directement l'éditeur :

```bash
make annotate-players \
  MEDIA=datas/train/HOU_LAL/G1/video.mp4 \
  CHECKPOINT=models/players/ebard_yolov8n.pt \
  PLAYER_CLASS=2 REFEREE_CLASS=3 PLAYER_DEVICE=cuda:0 \
  PLAYER_STEP=15 PLAYER_STOP=1842
```

`PLAYER_CLASS` et `REFEREE_CLASS` désignent les classes du checkpoint. Elles sont
converties en classes d'annotation `0=player` et `1=referee`. Omettre
`REFEREE_CLASS` conserve le mode joueur seul. Les frames déjà proposées, en cours
ou vérifiées restent intactes ; la commande complète uniquement les images non
annotées. Rien ne devient `verified` sans revue humaine.

La cible Make utilise `PREANNOTATION_PYTHON` (par défaut `PLAYERS_PYTHON`,
environnement ML) pour l'inférence, puis `PYTHON` (par défaut `.venv/bin/python`,
avec Tk) pour l'éditeur. Cela permet de travailler même si l'environnement
joueurs ne contient pas Tkinter. Sans `CHECKPOINT`, l'éditeur s'ouvre directement.

Pour préparer les **851 images train du plan**, sans ouvrir l'interface :

```bash
.venv-players/bin/python -m scripts.preannotate_players_plan \
  --plan /chemin/vers/players_annotation_queue.csv \
  --source datas --phase 1 \
  --checkpoint models/players/ebard_yolov8n.pt \
  --player-class 2 --referee-class 3 --device cuda:0 \
  --report reports/preannotation-train-nouvelle-execution.json
```

Le rapport doit être nouveau. Une relance préserve les propositions existantes.
Les autres phases restent sélectionnables explicitement avec `--phase` ; les
anciennes propositions val ne sont pas remplacées ni complétées automatiquement.

Pour une préannotation seule avec un autre modèle :

```bash
make preannotate-players \
  PYTHON=.venv-ball-torch/bin/python \
  MEDIA=datas/train/HOU_LAL/G1/video.mp4 \
  CHECKPOINT=/chemin/vers/detecteur-joueurs.pt \
  PLAYER_STEP=30 PLAYER_STOP=300 PLAYER_DEVICE=cpu
```

Cette phase fournit l'adaptateur **Ultralytics YOLO de détection**, sur checkpoint
`.pt` local existant. Aucun poids n'est téléchargé automatiquement. L'adaptateur
RF-DETR pour l'entraînement et le chargement autonome est décrit dans le
[guide de phase 4](TRAINING.md). `PLAYER_CLASS=0` sélectionne la classe source
à convertir en `player` ; choisir explicitement l'ID adapté aux poids. Un modèle
COCO « person » proposera aussi des arbitres et du public : reclasser les arbitres,
supprimer les boîtes du public.

Le CLI expose aussi `--confidence`, `--iou` et `--imgsz` :

```bash
.venv-ball-torch/bin/python -m training.players.annotator.preannotate --help
```

Les frames sont traitées une par une. Chaque proposition réussie est sauvegardée
immédiatement ; une interruption conserve ce travail. La relance complète les
frames encore non annotées et préserve toutes les propositions et corrections
existantes. Les sorties restent `proposed`, même sans détection, jusqu'à la revue.
Le SHA-256 des poids et les paramètres d'inférence sont conservés dans la provenance.
Les arguments correspondent à l'[API de prédiction Ultralytics](https://docs.ultralytics.com/modes/predict).

## Réutiliser un cache NBA

```bash
make preannotate-players MEDIA=datas/train/HOU_LAL/G1/video.mp4 \
  CACHE=/chemin/vers/video.players.MODELE.identifiant.json \
  CACHE_VIDEO_SHA256=SHA256_DE_LA_VIDEO PLAYER_STOP=300
```

L'import vérifie le format v1, le nom du média, les dimensions, la cadence, le
nombre total de frames, les indices continus du cache et les boîtes. Un cache
partiel ne produit aucune annotation pour sa partie manquante ; le rapport
affiche `not_cached` pour les frames sélectionnées non couvertes.

Si le cache fournit `meta.source_sha256`, ce digest doit correspondre à la vidéo.
Les anciens caches NBA n'ont pas cette information. Après vérification de leur
origine, fournir `CACHE_VIDEO_SHA256`, obtenu avec `sha256sum chemin/video.mp4`.
**Cette option déclare une association vérifiée par l'opérateur ; elle ne prouve
pas rétroactivement que le cache a été calculé sur ce contenu.** La provenance
enregistre `operator_asserted`, l'empreinte du cache, celle de la vidéo et le
modèle déclaré. Sans digest embarqué ni association explicite, l'import est refusé.

## Exporter les annotations locales en YOLO et COCO

```bash
make export-players
make verify-players-export
# Racine, sortie et configuration alternatives :
make export-players PLAYERS_ROOT=/chemin/datas PLAYERS_OUTPUT=/chemin/exports/players \
  PLAYERS_EXPORT_CONFIG=training/players/configs/players_export.yaml
```

Les deux formats sont toujours générés ensemble. Seules les frames marquées
`verified` sont retenues, y compris celles qui ne contiennent aucune boîte.
Les frames absentes, `unannotated`, `proposed` ou `in_progress` sont exclues.
L'export échoue si aucune frame vérifiée n'est disponible. Il ne valide aucune
prédiction et ne modifie aucun sidecar. La complétude visuelle des annotations
reste une décision humaine lors de la vérification de la frame.

Chaque média doit être sous `train/`, `val/` ou `test/`, avec un `match_id`
explicite dans son sidecar, `.meta.yaml` ou `match.yaml`. L'audit examine **tous
les splits**, y compris ceux exclus de l'export. Il refuse les matchs et contenus
identiques répartis entre splits, les identités contradictoires, les sidecars
invalides ou désynchronisés du média et les chemins symboliques dans les sources.
Les pixels identiques après décodage sont également refusés entre les frames
exportées de splits différents ; cela ne détecte pas des quasi-doublons visuels.
`split_by_venue: true` impose une salle connue pour chaque média audité et des
salles distinctes entre splits. Les dossiers `nba_cache/` sont ignorés.

La [configuration](configs/players_export.yaml) sélectionne `train` et `val` par
défaut. Inclure `test` uniquement pour constituer un export d'évaluation dédié,
une fois le protocole fixé ; aucun rééquilibrage ou déplacement de split n'est fait.
Un split sélectionné vide est conservé et signalé dans `stats.json` ; cela ne
garantit pas qu'un futur entraînement acceptera ce split vide.

Les images sont conservées aux dimensions natives en PNG par défaut. Le JPEG,
le redimensionnement `stretch` ou `letterbox` sont configurables. L'index
`frames.jsonl` garde les boîtes source, les boîtes exportées et les facteurs et
marges réels après arrondi des dimensions. Les coordonnées XYXY suivent les bords
des pixels ; les boîtes restent en flottants. YOLO emploie les classes `0` (joueur)
et `1` (arbitre) avec des centres/tailles normalisés ; COCO emploie les catégories
`1` et `2` respectivement, avec des boîtes XYWH en pixels. Les images sont encodées
une seule fois puis copiées à l'identique dans
les deux formats ; les négatifs ont un label YOLO vide et aucune annotation COCO.
Les recettes actuelles de détection et d'évaluation restent centrées sur les joueurs :
leurs vues privées filtrent les arbitres de l'export à deux classes.

```text
exports/players/
  current -> .generations/<identifiant>/
  yolo -> current/yolo/
  coco -> current/coco/
  .generations/<identifiant>/
    yolo/  # images/<split>/*, labels/<split>/*.txt, data.yaml
    coco/  # images/<split>/*, annotations/<split>.json
```

Chaque format contient aussi `manifest.json`, `config.resolved.yaml`,
`frames.jsonl`, `stats.json` et `split_audit.json`. Les chemins COCO sont relatifs
à la racine du format. Chaque dossier de format résolu est autonome et déplaçable.
Le manifeste inventorie tous les fichiers avec SHA-256, les sources et leurs
annotations, la provenance, les paramètres, les versions logicielles et les
empreintes du code. `dataset_id` est commun aux formats ; `export_id` inclut les
artefacts du format. Les identifiants ne dépendent ni de l'heure ni de la racine
absolue. Ils changent si les entrées auditées, le code, l'environnement enregistré
ou la configuration changent. Reproduire exactement les octets exige le même
environnement de décodage/encodage ; le manifeste conserve aussi l'empreinte de
la configuration de compilation OpenCV.

La publication utilise un verrou Linux, un dossier temporaire et une bascule
atomique de `current` **après vérification des deux formats**. Une erreur avant
cette bascule conserve la génération précédente. Les sorties étrangères sont
refusées, et les sources sont relues par empreinte avant publication pour détecter
les changements pendant l'export. Les anciennes générations sont conservées ;
aucune purge automatique. Cette protection concerne les erreurs de processus,
pas une garantie de durabilité après panne matérielle.

Pour lire plusieurs fichiers ou formats pendant qu'un autre export peut être
publié, résoudre `current` une fois et conserver ce chemin de génération :

```python
from pathlib import Path
generation = Path("exports/players/current").resolve(strict=True)
yolo_root, coco_root = generation / "yolo", generation / "coco"
```

`make verify-players-export` fait cette résolution, puis contrôle les empreintes,
les identifiants, les fichiers absents/supplémentaires, les images, la géométrie,
les labels, les statistiques et l'accord entre formats. Il fonctionne sans les
médias sources. Les empreintes servent à détecter les altérations ; elles ne sont
pas une signature attestant une revue humaine.

Les sidecars restent sous `datas/`, déjà suivi par `datas.dvc`. Après une session
d'annotation et fermeture de l'éditeur, leur versionnement local est explicite :

```bash
.venv/bin/dvc status
.venv/bin/dvc add datas
# Examiner puis committer datas.dvc avec le code et la configuration correspondants.
```

Cette commande actualise la référence et le cache DVC local. L'export ne lance
ni `dvc add`, ni `dvc push`, ni transfert réseau. Les exports restent régénérables,
ignorés par Git et hors de `datas.dvc`. Voir [les données](../../docs/DATA.md).

## Tests

```bash
make test-players
make test-ball
# Fenêtre Tk réelle, sur données synthétiques :
PLAYERS_GUI_TESTS=1 .venv/bin/python -m unittest training.players.tests.test_annotation.GuiTests -v
# Inférence CPU réelle, sans entraînement :
PLAYERS_MODEL_TESTS=1 PLAYERS_CHECKPOINT=/chemin/vers/yolo.pt \
  .venv-ball-torch/bin/python -m unittest training.players.tests.test_annotation.RealModelTests -v
# Infrastructure MLflow de phase 1, base temporaire :
make test-players-mlflow PYTHON=.venv-ball-torch/bin/python
```

Les tests habituels utilisent des médias synthétiques temporaires. Ils vérifient
les corrections, les négatifs explicites, les conflits de sauvegarde, la protection
des frames revues et les caches incomplets. Les tests
GUI, inférence réelle et MLflow sont séparément activables. Les tests d'export
utilisent uniquement des images et vidéos synthétiques : cohérence YOLO/COCO,
redimensionnements, fuites, intégrité, déterminisme, modifications concurrentes
et échecs avant publication.

Validation de la phase 2 (22 septembre 2026) : test Tk réel et inférence CPU avec
un checkpoint YOLO26 local passés ; non-régression balle : 56 tests passés,
13 intégrations optionnelles désactivées. Aucun entraînement ou test GPU lancé.

Validation de la phase 3 (23 septembre 2026) : 31 nouveaux tests d'export passés ;
suite joueurs : 60 tests passés et 6 intégrations optionnelles désactivées ;
suite balle : 56 tests passés et 13 intégrations optionnelles désactivées.
Le CLI d'export et la vérification sans sources sont testés sur fixtures.
Aucune donnée réelle modifiée, aucun entraînement ni transfert DVC lancé.

Le 23 septembre 2026, l'import de datasets externes a été retiré du périmètre à la
demande de l'utilisateur. L'annotation locale et la préannotation par modèle ou
cache NBA restent disponibles.
