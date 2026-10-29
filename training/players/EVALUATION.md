# Évaluation commune des joueurs et arbitres — phase 5

L'évaluation utilise les exports vérifiés de phase 3 et les adaptateurs YOLO26,
YOLOv8 et RF-DETR. Elle ne crée ni annotations, ni entraînements, ni transfert
DVC. Les métriques natives ne participent jamais au classement. Les sorties de
modèles préentraînés restent des prédictions, même sur une image sans annotation.
Le [parcours de phase 6](PIPELINE.md) ajoute l'orchestration et partage cette
fonction de prédiction avec la vidéo et les candidats du registre MLflow.
Les [contrôles communs du POC](../../docs/EVALUATION.md) documentent les suites
offline et les règles de promotion utilisées en CI et dans le registre.

## Commandes

Utiliser `.venv-players`, installé avec le lock décrit dans [TRAINING.md](TRAINING.md).
`pycocotools==2.0.11`, Matplotlib et psutil, déjà présents dans ce lock, sont
maintenant des dépendances directes de `requirements.in`. Aucun environnement
balle n'est modifié.

Copier une recette, puis renseigner les poids locaux, leur provenance `reference`,
l'export, la classe source et le matériel :

```bash
make evaluate-players CONFIG=training/players/configs/evaluate_yolo.yaml
make evaluate-players CONFIG=training/players/configs/evaluate_yolov8.yaml
make evaluate-players CONFIG=training/players/configs/evaluate_rfdetr.yaml
# Remplacer seulement les poids d'une recette :
make evaluate-players CONFIG=/recette.yaml CHECKPOINT=/poids/locaux.pt
# Sans Make :
.venv-players/bin/python -m training.players.evaluation evaluate --config /recette.yaml
```

Aucun entraînement personnalisé n'est nécessaire. Pour les poids COCO testés,
utiliser `source_class: 0` pour YOLO et `source_class: 1` pour RF-DETR ; pour les
checkpoints joueurs entraînés ici, utiliser `0`. La recette YOLOv8 E-BARD utilise
`source_class: 2` (joueur) et `referee_source_class: 3` (arbitre) dans le même
checkpoint. YOLO26 et RF-DETR ajoutent le checkpoint YOLOv8 indiqué par
`referee_weights`; `null` signifie que les deux classes viennent du checkpoint
principal. Un modèle « person » ne distingue pas automatiquement joueurs,
public, arbitres et banc ; une boîte joueur qui recouvre une boîte arbitre de
score ≥ 0,25 à IoU ≥ 0,6 est donc supprimée. Les poids importés exigent
une `reference` non vide décrivant leur véritable origine. Les checkpoints complets
joueurs fournissent leur run d'entraînement et son contrat, dont le budget et le
caractère smoke. Les poids doivent être de confiance (chargement PyTorch/pickle).

Les chemins relatifs des recettes se résolvent depuis la racine du dépôt. Le CLI
résout la génération d'export une seule fois et en vérifie l'intégrité avant et
après l'exécution. Les formats COCO et YOLO sont acceptés : ils possèdent le même
index de frames. Seules les frames vérifiées du split choisi sont évaluées, y
compris les négatifs explicites. Une sélection vide est refusée. `max_images`
limite aux premières frames de l'index, en enregistrant exactement cette sélection.

## Métriques et coordonnées

Les images d'export, éventuellement redimensionnées, sont les entrées des modèles.
Chaque adaptateur restitue des XYXY dans les pixels de cette entrée. L'évaluateur
inverse ensuite les facteurs et marges de `transform`, coupe aux dimensions
source et supprime les boîtes dégénérées dans les marges. Les prédictions JSONL,
les appariements, les aires et les métriques utilisent les coordonnées source.
Les exemples PNG sont dessinés sur les images d'export avec la transformation
inverse de cet affichage ; les médias originaux ne sont pas nécessaires.

- AP@50 et AP@50:95 : `pycocotools.COCOeval`, type `bbox`, séparément pour
  `player` et `referee` ; le global est la moyenne des classes ayant une GT.
  IoU 0,50 à 0,95 par pas de 0,05, interpolation sur 101 points de rappel. La
  [référence COCO officielle](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
  définit ce calcul. Une absence de vérité terrain produit `null`, jamais un AP fictif.
- `score_floor: 0.001` pour collecter les candidats AP ; un plancher supérieur à
  0,01 est refusé. Les frameworks filtrent nativement strictement au-dessus du
  plancher. La courbe reste limitée par ce plancher et leurs candidats natifs :
  ce n'est pas l'ensemble théorique de toutes les boîtes possibles.
- `max_detections: 100` par défaut, configurable entre 10 et 300 : limite native
  YOLO, tri stable et limite après filtrage de classe RF-DETR, puis après retour
  aux coordonnées source. RF-DETR sélectionne aussi ses candidats en interne
  avant filtrage de classe ; les deux architectures n'ont pas le même générateur
  de candidats. Aucun NMS supplémentaire n'est appliqué par l'évaluateur.
- Précision/rappel/F1 : `score_threshold: 0.25`, `iou_threshold: 0.5` par défaut.
  Les prédictions sont traitées par score décroissant ; chacune prend la vérité
  terrain libre avec la plus forte IoU **supérieure ou égale** au seuil. À égalité,
  l'ordre de l'index GT départage. Une GT ne peut être prise qu'une fois : les
  doublons sont des FP. Un score égal au seuil d'exploitation est conservé.
- `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `F1 = 2TP/(2TP+FP+FN)`.
  Une précision sans prédiction vaut `null`. Sans GT, rappel et F1 valent `null` ;
  avec GT mais sans détection, rappel et F1 valent zéro. Les comptes restent visibles.

Changer le seuil d'exploitation ne change pas l'AP. Aucun seuil d'affichage vidéo
n'intervient. Le fichier `curves.json` et les CSV associés conservent les courbes
PR COCO et précision/rappel/F1 selon le score, également tracées dans `curves.svg`.
Aucun seuil optimal n'est choisi automatiquement.

Les groupes comprennent aussi une ligne par classe, puis match, clip (`source_id`), salle renseignée ou inconnue,
et tailles COCO selon l'aire **source** : small jusqu'à 32², medium entre 32² et
96², large à partir de 96². Les bornes sont inclusives comme dans COCO. Les GT
hors intervalle et les prédictions appariées à celles-ci sont ignorées ; les
prédictions non appariées hors intervalle le sont également pour cette tranche.

L'occultation est rapportée en **groupes d'images** contenant au moins une GT
`occluded=true`, `false` ou inconnue. Toutes les GT de l'image sont conservées.
Ces groupes peuvent se chevaucher ; ce ne sont pas des AP conditionnels par
objet occulté. Les images négatives restent dans le global et les groupes
match/clip/salle, mais n'appartiennent à aucun groupe d'occultation. Cette limite
évite de transformer les joueurs visibles exclus d'un groupe en faux positifs.
Les groupes sans support portent des métriques nulles et leurs effectifs réels.

## Vitesse et mémoire

Le protocole actuel accepte **FP32 uniquement**, batch 1, sans TF32, compilation,
TensorRT ni optimisation RF-DETR. Les recettes proposées utilisent la même
résolution carrée 640 ; les essais techniques courts utilisent 128. YOLO réalise
son letterbox natif (`rect=False`), RF-DETR son redimensionnement et sa normalisation
natifs avec conversion BGR→RGB. La résolution d'export est conservée dans la
sélection. Ces prétraitements propres aux architectures sont explicités, pas
remplacés par une transformation incompatible avec leurs poids.

`protocol.json` décrit machine, CPU, GPU et pilote quand disponible, versions
PyTorch/CUDA/cuDNN et des dépendances, précision, résolution, threads, score,
IoU, limites, batch, préchauffage et répétitions. Le protocole de mesure est :

1. Charger le modèle et produire les prédictions de qualité ; ces opérations
   initiales ne font pas partie des temps publiés.
2. Préchauffer `warmup` fois sur la première image (5 par défaut), synchroniser
   CUDA après chaque appel, puis réinitialiser les pics mémoire CUDA.
3. Parcourir toutes les images `repeats` fois (3 par défaut), séquentiellement.
   Synchroniser le GPU avant et après chaque appel chronométré avec `perf_counter`.
   Mesurer séparément l'adaptateur complet (prétraitement, transferts, forward,
   post-traitement, conversion CPU) et le même appel avec décodage OpenCV.
4. Faire une seconde passe avec hooks autour de chaque module PyTorch et synchronisation
   avant/après le forward. Cette instrumentation ne pénalise pas les temps de la
   première passe. Le forward YOLO englobe son `AutoBackend` ; RF-DETR utilise le
   module de détection avant son post-traitement public. Quand le modèle arbitre
   est séparé, son temps forward est ajouté au temps du modèle joueur.
5. Publier médiane, P95 (percentile NumPy), débit séquentiel `N/somme(temps)` et
   tous les échantillons. Le débit forward est un débit équivalent, pas une mesure
   de service concurrent. Les écritures des rapports, l'inversion de géométrie
   d'export et le chargement initial sont exclus.

Mémoire : pics PyTorch alloués et réservés pendant les passes, après préchauffage,
y compris le modèle résident ; ils n'incluent pas toutes les allocations du pilote
ou d'autres processus. Sur CPU, seul le pic RSS **échantillonné** du processus est
fourni, pas une mesure exacte de la mémoire du modèle. Sur CPU les champs GPU sont
`null`. Le cache disque est chaud ; ne pas comparer à un démarrage à froid.

`video_pipeline` reste explicitement `null` : cette phase mesure des images
exportées. Décodage d'une vidéo, encodage, annotation et débit vidéo de bout en
bout sont disponibles dans le parcours vidéo de phase 6 et ne sont pas mesurés ici. Il n'y a aucune
promesse de latence vidéo à partir du temps forward.

## MLflow, artefacts et comparaison

Chaque évaluation crée un run explicite `players-evaluation` (URI locale
`sqlite:///mlflow.db` par défaut ; `MLFLOW_TRACKING_URI` prend priorité). Elle
conserve le lien vers l'entraînement ou la référence importée, les SHA-256 des
deux checkpoints, l'export, la sélection, le code et l'environnement. Un échec/interruption
conserve les artefacts disponibles et le statut FAILED/KILLED du tracker commun.

Sous `runs/players/evaluation/<run_id>/` :

- `recipe.json` réutilisable avec `--config`, `config.resolved.yaml`, manifeste,
  environnement, code, sélection, référence et protocole ;
- `predictions.jsonl` avec une ligne par image, même vide, et identité source ;
- `metrics.json` / `metrics.csv`, `curves.json` / CSV / SVG ;
- `review.csv` : chaque TP/FP/FN avec indices GT/prédiction, score, IoU et boîtes ;
- `examples/` : au plus `max_examples` images avec erreurs, TP vert, FP rouge,
  FN magenta ; `examples.json` indique le nombre disponible et les limites ;
- `performance.json`, `result.json`, `report.html` et `summary.json`.

Ces fichiers sont aussi journalisés dans MLflow. Les exemples ne deviennent
jamais des sidecars vérifiés. Aucun choix automatique de modèle ni promotion.

```bash
make compare-players RUNS="runs/players/evaluation/<run-yolo> runs/players/evaluation/<run-rfdetr>" \
  REPORT_OUTPUT=runs/players/comparaison-val
# Le dossier de sortie doit être nouveau ; espaces dans les chemins : utiliser le CLI.
.venv-players/bin/python -m training.players.evaluation compare \
  /evaluation/run-yolo /evaluation/run-rfdetr --output /nouveau/rapport
```

Le rapport CSV/JSON/HTML classe par AP@50:95 puis AP@50 ; les égalités partagent
le rang. Il crée des **groupes de classement indépendants** pour chaque identité
de données, sélection et protocole exacts, y compris matériel, versions, code,
résolution, précision, seuils et procédure de mesure. Deux groupes distincts ne
reçoivent jamais de rang commun. Une différence de résolution empêche donc un
classement direct ; elle reste affichée. Les budgets d'entraînement, seeds,
préentraînements, classes source et liens vers les runs restent visibles.

Les smoke tests (évaluation **ou checkpoint d'entraînement**) sont exclus par
défaut, ainsi que les runs non terminés ou sans support GT. Pour vérifier le
rapport sur fixtures uniquement : `--include-smoke` ou `INCLUDE_SMOKE=1`.
Les différences de budget/préentraînement restent des facteurs expérimentaux ;
l'identité de protocole ne les annule pas. Aucun essai multi-seeds n'est lancé.

## Réserver le test final

Choisir modèle, poids, données, seuils et protocole à partir de `val`. Préparer
une recette `split: test`, sans `max_images`, `purpose: evaluation`, puis :

```bash
.venv-players/bin/python -m training.players.evaluation evaluate \
  --config /recette-test.yaml --freeze-recipe /recette-test.frozen.json
# Examiner et versionner ce fichier AVANT de lire les résultats test.
# Dans recette-test.yaml : frozen_recipe: /recette-test.frozen.json
.venv-players/bin/python -m training.players.evaluation evaluate --config /recette-test.yaml
```

Le gel ne lance aucune inférence. Il lie la recette aux empreintes des deux poids,
données, sélection et code d'évaluation. L'exécution refuse un gel absent ou
modifié, les sélections test partielles et les checkpoints smoke. Déplacer les
poids/export sans changer leur contenu reste possible. Modifier le code, les
poids ou les seuils exige un nouveau gel explicite. Le fichier est une barrière
reproductible, pas une preuve cryptographique que personne n'a déjà consulté le
test. Ne pas utiliser ses courbes pour régler les seuils ; elles décrivent la
recette finale et n'en choisissent aucun.

## Validation courte et limites constatées

```bash
make test-players-evaluation
make test-players
make test-ball
PLAYERS_EVALUATION_TESTS=yolo PLAYERS_TEST_WEIGHTS=/poids/yolo26n.pt \
  .venv-players/bin/python -m unittest training.players.tests.test_evaluation.RealEvaluationTests -v
PLAYERS_EVALUATION_TESTS=rfdetr PLAYERS_TEST_WEIGHTS=/poids/rf-detr-nano.pth \
  PLAYERS_TEST_DEVICE=cuda:0 \
  .venv-players/bin/python -m unittest training.players.tests.test_evaluation.RealEvaluationTests -v
```

`PLAYERS_TEST_OUTPUT` conserve les fixtures et artefacts, sinon ils sont
temporaires. Les intégrations utilisent deux images val synthétiques, résolution
128, FP32, un préchauffage et deux répétitions : seulement quatre échantillons
par périmètre chronométré. Elles vérifient le fonctionnement des mesures et de
MLflow ; leur P95 n'est pas une estimation robuste et leurs AP ne mesurent pas
la qualité sur un match. Les tests analytiques couvrent TP/FP/FN, doublons,
images vides, seuils inclusifs, aires, occultation, transformations stretch et
letterbox (y compris marges), égalité des métriques entre les adaptateurs,
comparaisons incompatibles, exclusion smoke et gel du test.

RF-DETR signale que le modèle n'est pas optimisé pour l'inférence ; c'est bien
le chemin natif FP32 décrit ici. Il émet aussi des avertissements de dépréciation
TorchScript et de configuration du backbone lors du chargement des poids locaux.
Aucune donnée réelle annotée n'a été utilisée dans ces essais, aucun entraînement
n'a été lancé, aucune prédiction n'a été validée comme annotation et aucun DVC
add/push n'a été effectué. Il reste à constituer la vérité terrain humaine pour
comparer réellement la qualité des deux familles.

Résultats techniques du 24 septembre 2026 : **83 tests joueurs réussis et 8
intégrations optionnelles désactivées** dans `.venv-players` ; **78 réussis et 13
désactivés** dans l'environnement léger `.venv` (COCO optionnel absent).
Non-régression balle : **56 réussis, 13 intégrations désactivées**. Les quatre
intégrations d'évaluation préentraînée YOLO26 Nano / RF-DETR Nano, sur CPU et
RTX 5080, ont passé séparément, sans entraînement.

Pour ces seules fixtures (128×128, FP32, batch 1, quatre échantillons), l'appel
complet à l'adaptateur a donné :

| Modèle | Matériel | Médiane / P95 (ms) | Débit séquentiel (images/s) | Pic CUDA alloué (MiB) |
| --- | --- | ---: | ---: | ---: |
| YOLO26 Nano | Ryzen 9 9900X | 3,90 / 3,93 | 257,3 | — |
| RF-DETR Nano | Ryzen 9 9900X | 23,28 / 23,37 | 42,9 | — |
| YOLO26 Nano | RTX 5080 | 2,88 / 3,17 | 341,6 | 179,6 |
| RF-DETR Nano | RTX 5080 | 4,23 / 4,27 | 237,3 | 133,8 |

Ces chiffres documentent un essai technique très court, pas un benchmark de
production ni une recommandation de modèle. Les valeurs forward, avec décodage,
mémoire réservée et RSS, tous les échantillons et les identifiants MLflow sont
conservés localement dans `runs/players/phase5-validation/validation.summary.json`
et les sous-dossiers `final-*/runs/<run_id>/`.

Une différence de threads OpenCV a été constatée pendant la validation : l'import
Ultralytics les modifie. L'évaluateur fixe maintenant **OpenCV à un thread** pour
les deux familles et PyTorch à `cpu_threads` (4 dans ces essais). La comparaison
technique a confirmé deux groupes séparés (CPU / GPU), chaque paire ayant les
mêmes données et exactement le même protocole. Le rapport par défaut exclut les
quatre runs smoke ; `comparison-technical/` les montre uniquement avec l'option
explicite. Les commandes CLI de comparaison et de gel sans inférence sont testées.
