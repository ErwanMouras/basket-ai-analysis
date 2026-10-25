# Orchestration locale, vidéo et registre — phase 6

Cette phase relie les contrats existants. L'orchestrateur exécute validation,
export YOLO/COCO, entraînement facultatif, évaluation commune et comparaison.
L'inférence vidéo et les opérations de registre sont des commandes explicites
séparées. Aucun service web, maillot, déploiement ou transfert DVC n'est ajouté.
Depuis le 26 septembre, l'inférence vidéo propose aussi un
[suivi optionnel BoT-SORT/ByteTrack](TRACKING.md), avec `track_id` par observation
confirmée. Ces IDs ne constituent pas une identité joueur sur tout le match.
La [démo de phase 7](VALIDATION.md) exécute ce parcours sur fixtures synthétiques
avec poids aléatoires locaux, tests de corruption et reprise, puis registre et
vidéo. Les [résultats](VALIDATION_RESULTS.md) indiquent les tests réellement exécutés.

## Pipeline YAML

Utiliser l'environnement `.venv-players` de [TRAINING.md](TRAINING.md). Copier
[configs/pipeline.yaml](configs/pipeline.yaml) et renseigner les recettes et poids
locaux. Les chemins relatifs partent de la racine du dépôt, comme en phases 4/5.

```yaml
schema_version: 1
source: datas
output: runs/players/mon-parcours
export_config: training/players/configs/players_export.yaml
include_smoke: false
trials:
  - name: yolo
    train: /recettes/train-yolo.yaml
    evaluate: /recettes/evaluate-yolo.yaml
  - name: rfdetr
    train: /recettes/train-rfdetr.yaml
    evaluate: /recettes/evaluate-rfdetr.yaml
```

```bash
make pipeline-players CONFIG=/recettes/pipeline.yaml
# Relancer exactement la même commande reprend le parcours.
.venv-players/bin/python -m training.players.orchestration.pipeline \
  --config /recettes/pipeline.yaml
```

Chaque essai a un nom unique. `train: null` (ou absent) évalue directement les
poids préentraînés de la recette d'évaluation. Avec entraînement, le pipeline
injecte l'export figé et le checkpoint `best` dans l'évaluation ; famille/variante
doivent correspondre et `source_class` doit valoir `0`. Les recettes d'entraînement
conservent leur budget, seed, matériel, précision et mode. Les destinations des
runs sont placées sous le parcours, sans modifier les recettes d'origine.

Chaque étape utilise **un processus séparé**, toujours **séquentiellement**, sur
CPU comme sur GPU. Cela évite le partage involontaire des états des frameworks
et les essais GPU concurrents dans un même parcours. Un verrou empêche deux
orchestrateurs d'écrire le même parcours. Deux parcours dans des dossiers différents
peuvent cependant utiliser le même GPU : il n'y a pas d'ordonnanceur global.

Le pipeline de sélection utilise `val` et refuse les exports contenant `test`.
Le test final reste accessible uniquement par la commande d'évaluation avec une
recette figée, décrite dans [EVALUATION.md](EVALUATION.md). Le classement reprend
les mêmes groupes de protocoles et l'exclusion smoke de phase 5.

## Progression et reprise

`output/progress.json` est remplacé atomiquement. Il contient l'étape, son statut,
le run MLflow courant, l'epoch et les epochs terminées, le batch et son total,
le temps écoulé de la tentative de parcours, une estimation lorsqu'une epoch
est mesurée, les sorties, erreurs et l'historique des tentatives. Les compteurs
connus sont alimentés par les callbacks des deux entraîneurs ; une valeur inconnue
reste `null`. L'estimation par epoch est indicative, notamment après une reprise.

Sous `attempts/<étape>/<identifiant>/`, consulter :

- `job.json` : configuration effective, entrées et empreinte ;
- `live.json` : événement atomique du processus enfant ;
- `worker.log` : sortie complète, sans parser la barre de progression du framework ;
- `receipt.json` : résultat terminé, racines d'artefacts et SHA-256 des fichiers ;
- `failure.json` : erreur, run et checkpoints éventuellement disponibles ;
- `runs/` ou `report/` : les artefacts de la phase concernée.

Une étape terminée est réutilisée seulement si sa configuration, ses entrées,
le code, l'environnement enregistré **et tous ses artefacts** correspondent aux
empreintes. Un fichier absent, supplémentaire ou modifié invalide sa réutilisation.
Les liens vers les images d'export sont aussi vérifiés. Les générations exportées
restent figées par chemin ; changer le lien public `current` ne change pas celles
utilisées par un parcours terminé. L'audit relit les sources pour détecter également
l'apparition d'annotations ou métadonnées auparavant absentes.

Les étapes invalidées produisent une nouvelle tentative et conservent les
anciennes. Seules les étapes dépendant des entrées modifiées sont recalculées ;
changer uniquement le seuil d'évaluation ne relance pas l'entraînement. Une
interruption du parent après l'écriture du reçu peut être récupérée sans nouveau
run. Si son enfant travaille encore, le verrou de cet enfant bloque une seconde
exécution plutôt que de lancer un doublon.

Après interruption d'un entraînement, le pipeline utilise `last` et son `best`
compagnon uniquement si leurs SHA-256 enregistrés correspondent encore. Il
active alors la **reprise stricte de phase 4**, qui vérifie données, budget,
optimiseur, scheduler, EMA, RNG, code et environnement. Une reprise crée son
propre run MLflow relié au parent. Elle recommence l'epoch interrompue, pas
nécessairement le batch précis. Elle ne devient jamais silencieusement un
fine-tuning. Pour changer la recette ou le budget, fournir explicitement une
recette `mode: finetune` avec les poids souhaités.

Sans checkpoint vérifiable après création d'un run, la relance s'arrête et demande
une décision explicite. Après inspection :

```bash
.venv-players/bin/python -m training.players.orchestration.pipeline \
  --config /recette.yaml --restart-incomplete
# Équivalent Make : RESTART_INCOMPLETE=1
```

Cette option redémarre la recette d'origine ; elle peut donc refaire du travail.
Un checkpoint ayant atteint son budget sans reçu final reste ambigu : la reprise
stricte refuse de le prolonger. Les reçus attestent des empreintes locales, pas
une signature cryptographique. Un SIGKILL ne permet ni finalisation MLflow ni
sauvegarde : inspecter les runs éventuellement restés RUNNING. Un Ctrl-C est
transmis au groupe du processus enfant ; après 20 secondes sans sortie, il est
arrêté de force. Les fichiers déjà publiés ne sont pas supprimés.

## Inférence vidéo sans annotations

Les [numéros de maillot](../jersey/README.md) s’ajoutent avec `make predict-jersey`.
L’OCR est limité dans le temps et fonctionne sur CPU par défaut.

Pour ajouter les squelettes aux boîtes et aux IDs, voir [POSE.md](POSE.md) :
`make setup-players-pose`, puis `make pose-players VIDEO=... CHECKPOINT=... OUTPUT=...`.
Cette commande active le suivi et RTMPose-M dans le même parcours vidéo.

```bash
make predict-players CONFIG=training/players/configs/predict_video.yaml \
  VIDEO=/videos/clip.mp4 CHECKPOINT=/poids/yolo26n.pt
# Ou :
.venv-players/bin/python -m training.players.predict --config /recette-video.yaml
```

Le répertoire de sortie doit être nouveau. Les champs famille, variante,
classe source, poids arbitre, résolution, device et précision suivent l'évaluation.
La recette fournie utilise YOLOv8 E-BARD pour ajouter les arbitres (`class_id: 1`)
aux joueurs (`class_id: 0`). Le suivi, la pose, les maillots et les positions terrain
continuent de traiter les joueurs ; les arbitres sont conservés dans les prédictions,
les traces et la vidéo avec `track_id: null`. La fonction
`training.players.inference.predict_image` est partagée avec l'évaluateur :
prétraitements natifs, FP32, entrée carrée, filtrage et plafonnement des boîtes,
coordonnées XYXY source. La vidéo n'a pas de transformation d'export préalable.
Les réglages GPU/threads sont enregistrés comme pour l'évaluation.

Une seule frame décodée et ses détections sont traitées à la fois ; le JSONL est
écrit immédiatement, sans accumuler la vidéo ou toutes ses boîtes en mémoire.
`score_floor` gouverne l'export des prédictions ; `score_threshold` ne filtre que
l'affichage. La sortie contient `predictions.jsonl`, `annotated.mp4`, provenance,
`progress.json` et `result.json`. Le SHA-256 source, l'indice de frame et le temps
nominal `index/fps` accompagnent chaque ligne, même sans détection.

La recette vidéo fournie active maintenant BoT-SORT. Le suivi ajoute
`tracks.jsonl` et `tracking.json` et affiche les IDs sur la vidéo ; le format des
détections brutes reste identique. `--tracker none` ou `tracking.enabled: false`
désactive ce suivi. Les recettes antérieures sans section `tracking` restent en
détection seule. Voir [TRACKING.md](TRACKING.md) pour les seuils et les limites.

La sortie MP4 utilise `mp4v`, conserve les dimensions paires et le FPS déclaré,
et ne conserve pas l'audio. Les dimensions impaires ou changeantes sont refusées
pour éviter une coupe silencieuse. Les timestamps sont nominaux : les vidéos à
framerate variable ne sont pas reproduites fidèlement. `max_frames` permet un
essai borné. Un échec de décodage avant le compte de frames annoncé est signalé.

Une interruption conserve `predictions.partial.jsonl` et `annotated.partial.mp4`,
avec statut KILLED/FAILED ; aucune sortie partielle ne porte le nom final. Il
n'y a pas de reprise de l'encodage au milieu d'une vidéo : recommencer dans un
nouveau dossier. Aucune annotation humaine n'est requise ni écrite.

Le débit vidéo `end_to_end_fps` inclut chargement, décodage, inférence, dessin,
encodage et écritures ; il n'a pas de préchauffage séparé. C'est une mesure du
parcours, distincte des latences modèle/adaptateur de phase 5. Les clips très
courts sont dominés par le démarrage et ne constituent pas un benchmark.

## Candidats MLflow et alias

Le modèle s'appelle **`players-detector`**. `MLFLOW_TRACKING_URI` prend priorité
sur `mlflow_uri` des recettes ; SQLite local reste le défaut. Aucun modèle n'est
enregistré ou promu automatiquement par le classement.

```bash
make register-players CONFIG=training/players/configs/register.yaml
```

La recette `action: register` désigne un dossier d'évaluation et le checkpoint
exact. La commande vérifie son SHA-256, les identités des données et les artefacts
canoniques du run `players-evaluation` terminé dans le même store. Elle crée un
candidat MLflow pyfunc avec poids, protocole, liens entraînement/évaluation,
manifestes, code Python joueurs/commun et dépendances épinglées. Les données
images ne sont pas copiées dans le modèle ; leurs identifiants et les liens
MLflow sont conservés. L'expérience de packaging est `players-registry`.

Une nouvelle invocation avec exactement le même candidat réutilise sa version.
L'identité inclut l'évaluation, les poids, le code et les dépendances ; une autre
évaluation crée donc un autre candidat, même si les poids sont identiques. Un
packaging terminé avant une interruption est réutilisé lors de la création de
version. Les runs échoués restent disponibles pour inspection.

Un candidat smoke peut être enregistré et rechargé pour vérification technique,
mais **ne peut pas être promu par alias**. La promotion `champion` exige désormais une politique de qualité et une baseline explicites ; voir [les contrats](../../docs/EVALUATION_AND_RELEASE.md). Les alias expérimentaux ne dépendent d'aucun seuil AP
automatique : c'est une décision humaine explicite.

```yaml
action: promote
alias: champion
quality_policy: configs/quality_gates.yaml
baseline_evaluation: /evaluation/baseline
version: "2"
expected_previous: "1"  # null uniquement si l'alias n'existe pas
reason: "Recette validée sur les matchs de validation"
mlflow_uri: sqlite:///mlflow.db
```

```bash
make register-players CONFIG=/recette-promotion.yaml
```

Le changement est journalisé dans `players-registry-history` avec acteur, raison,
version précédente, cible, intention avant écriture et résultat après contrôle.
Pour revenir de 2 à 1, promouvoir explicitement `version: "1"` avec
`expected_previous: "2"` et une raison de retour arrière. Une précondition périmée
est refusée ; une répétition vers la version déjà active ne crée pas un nouveau
changement. Les [alias MLflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/)
sont utilisés, sans anciens stages Staging/Production.

Les commandes locales prennent un verrou par store SQLite. Pour un serveur
partagé, le verrou coordonne seulement les commandes de ce poste ; les API
MLflow n'offrent pas de compare-and-swap atomique de l'alias. Une mutation externe
concurrente ou un arrêt entre écriture de l'alias et finalisation du journal peut
laisser une intention à examiner. Il ne s'agit pas d'un service distribué.

Rechargement Python dans l'environnement compatible :

```python
import mlflow
import numpy as np
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model("models:/players-detector/1")
# uint8 BGR : une image HWC ou un lot NHWC ; résultat : une liste par image.
boxes = model.predict(np.zeros((96, 128, 3), dtype=np.uint8))
```

Le pyfunc charge sur CPU. Pour la vidéo, remplacer `weights` par `null` et
renseigner `registry_uri: models:/players-detector@champion` (ou `/1`) et
`mlflow_uri`. L'alias est résolu une seule fois en version ; la famille, variante,
résolution et classe source viennent du candidat, le device reste choisi dans
la recette vidéo. Les poids sont téléchargés depuis les artefacts du store et
leur empreinte est vérifiée avant usage. Le bundle ne configure pas le pilote
CUDA : utiliser l'environnement compatible documenté.

## Tests courts

```bash
make test-players-pipeline
make test-players
make test-ball
PLAYERS_PIPELINE_TESTS=yolo PLAYERS_TEST_WEIGHTS=/poids/yolo26n.pt \
  PLAYERS_TEST_OUTPUT=runs/players/validation-yolo \
  .venv-players/bin/python -m unittest training.players.tests.test_pipeline.RealPipelineTests -v
PLAYERS_PIPELINE_TESTS=rfdetr PLAYERS_TEST_WEIGHTS=/poids/rf-detr-nano.pth \
  PLAYERS_TEST_OUTPUT=runs/players/validation-rfdetr \
  .venv-players/bin/python -m unittest training.players.tests.test_pipeline.RealPipelineTests -v
```

Ces intégrations créent quatre images train et deux val synthétiques, entraînent
sur deux epochs à résolution 128, interrompent après la première, puis reprennent
strictement. Elles évaluent, comparent, relancent sans nouveaux runs, invalident
un artefact de comparaison, enregistrent et rechargent un candidat, puis traitent
quatre frames vidéo sans annotations. Les fixtures de registre vérifient aussi
promotion explicite, retour arrière, précondition d'alias et refus des smoke.
Toutes les bases MLflow sont locales et isolées. `PLAYERS_TEST_DEVICE=cuda:0`
permet un essai GPU volontaire ; les sorties conservées exigent un nouveau chemin
pour une nouvelle intégration complète.

### Résultats techniques du 25 septembre 2026

Les deux parcours réels ont passé sur CPU (Ryzen 9 9900X), FP32, résolution 128,
avec quatre images train, deux images val et quatre frames vidéo synthétiques :

| Parcours | Entraînement | Réutilisation | Registre et vidéo |
| --- | --- | --- | --- |
| YOLO26 Nano | 1 epoch interrompue + 1 epoch après reprise stricte | Aucun nouveau run à la relance ; seul le rapport altéré est recalculé | Une version réutilisée à la seconde inscription ; pyfunc et vidéo rechargés |
| RF-DETR Nano | 1 epoch interrompue + 1 epoch après reprise stricte | Même contrôle ; changement du seuil relançant uniquement évaluation/comparaison | Une version réutilisée à la seconde inscription ; pyfunc et vidéo rechargés |

Chaque parcours a produit exactement deux runs d'entraînement (KILLED puis
FINISHED), avec lien de reprise. Le second contrôle de seuil RF-DETR a conservé
ces deux runs et créé uniquement une seconde évaluation. Les smoke sont exclus
du rapport par défaut et leur promotion a été refusée. Les fixtures de registre
ont vérifié une promotion 1→2 puis un retour 2→1 avec trois traces d'audit, ainsi
que le refus d'une précondition périmée et d'un smoke même si son tag est modifié.

Les tests de processus couvrent aussi l'arrêt du processus enfant sur Ctrl-C,
la récupération d'un reçu terminé après perte de la mise à jour du parent,
les fichiers absents/supplémentaires/modifiés, et une tentative de reprise échouée
qui ne doit pas masquer le run interrompu précédent. La vidéo interrompue après
deux frames conserve uniquement les fichiers partiels et son erreur.

Les traces locales sont dans `runs/players/phase6-validation/`, notamment
`yolo-cpu/validation.json`, `rfdetr-cpu/validation.json`, les bases SQLite,
les `pipeline/progress.json`, les reçus et vidéos. Ces fixtures vérifient le
fonctionnement ; elles ne mesurent pas la qualité sur de vrais matchs. Aucun
essai GPU de phase 6 ni entraînement long n'a été lancé. Les essais GPU des
phases précédentes ne remplacent pas une validation GPU de ce nouveau parcours.

Limites effectivement constatées : MLflow renvoie ici les numéros de version
comme des entiers, normalisés en chaînes dans les commandes d'alias. Il signale
également l'absence de version de `pip` dans l'environnement créé avec `uv` :
le `conda.yaml` généré laisse donc `pip` non épinglé ; les distributions du lock
fourni restent épinglées. RF-DETR émet ses avertissements de chargement de tête
à une classe et d'inférence non optimisée ; les tests de reprise/rechargement
passent sur ce chemin FP32. Aucune donnée réelle ni annotation existante n'a été
modifiée, et aucune commande DVC n'a été exécutée.

Bilan des suites après intégration : **92 tests joueurs réussis, 9 intégrations
optionnelles désactivées** dans `.venv-players` ; **86 réussis, 15 désactivés**
dans `.venv` (dont MLflow et COCO indisponibles). Non-régression balle : **56
réussis, 13 intégrations optionnelles désactivées**. Les deux parcours réels
ci-dessus ont été exécutés séparément avec leurs options explicites. `git diff
--check` ne signale pas d'erreur d'espacement.
