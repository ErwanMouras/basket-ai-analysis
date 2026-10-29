# Évaluation et exécution du POC

Les commandes de ce document n'entraînent aucun modèle et ne créent pas de vérité
terrain. Elles utilisent des poids existants ou des annotations déjà revues.

## Données et protocole

Séparer train, validation et test par match et vérifier les doublons. La
validation sert aux réglages ; réserver le test à une recette gelée avant la
lecture des scores. Publier les supports par match et leur couverture temporelle,
car des images voisines ne constituent pas des exemples indépendants.

Comparer les modèles sur les mêmes données, coordonnées source, matériel et
précision numérique. Séparer les temps d'adaptateur, de décodage et de pipeline.
Les protocoles spécialisés décrivent les seuils, le contexte et le gel des recettes :
[joueurs](../training/players/EVALUATION.md) et
[ballon](../training/ball/EVALUATION.md).

## Contrôles techniques

```bash
make check-poc CHECK_OUTPUT=runs/checks/nouveau-dossier
```

Le runner enlève les variables d'activation des intégrations d'entraînement.
Les suites, logs, nombres de tests et skips sont dans `summary.json`. Le workflow
GitHub reprend ce parcours ; son exécution distante exige une publication des
changements. Les tests offline ne valident pas la précision des modèles réels.

## Politique de sélection

`configs/quality_gates.yaml` impose des données, sélection et protocole identiques,
des évaluations terminées non-smoke, un support minimum, des tolérances de baisse
AP/rappel et un ratio maximum de latence P95. Les seuils sont provisoires et
documentés ; ils doivent être arrêtés sur val pour une version donnée.

```bash
.venv-players/bin/python -m mlops.quality \
  --baseline /evaluation/baseline --candidate /evaluation/candidat \
  --policy configs/quality_gates.yaml --output /nouveau/decision.json
```

Le code de sortie vaut 0 pour une acceptation, 2 pour un refus. Une valeur
manquante, NaN ou un protocole incompatible ne donne pas une réussite implicite.
Le rapport conserve les deux runs, checkpoints, politique et détails des contrôles.

La promotion `champion` du registre exige `quality_policy` et
`baseline_evaluation`. Le contrôle est recalculé sur le résultat candidat conservé
dans MLflow avant de modifier l'alias. Il est joint à l'audit d'une promotion
appliquée. Une raison et la version `expected_previous` restent obligatoires.
Les alias expérimentaux autres que `champion` peuvent être gérés manuellement.

La baseline de référence est un choix explicite. Pour un retour arrière
opérationnel, reprendre la baseline précédemment acceptée et sa référence
d'acceptation d'origine, avec une nouvelle raison ; ne pas présenter ce retour
comme une amélioration de qualité face à la dernière version.

## Preuve du cycle, sans entraînement

```bash
.venv-players/bin/python -m mlops.acceptance --output runs/acceptance/nouveau
```

Le programme utilise les poids locaux YOLO26 Nano et E-BARD, un export joueurs
vérifié et une nouvelle base MLflow dans le dossier de sortie. Il évalue les deux
modèles, enregistre les versions, applique/refuse les promotions, restaure la
version précédente et vérifie une prédiction du modèle restauré. Il n'altère pas
la base MLflow existante ni les annotations. Les scores restent ceux de val.

L'analyse intégrée accepte aussi `people.adapter: registry`, `registry_uri` et
`mlflow_uri` dans l'environnement joueurs complet. Une version explicite est
résolue et ses empreintes sont conservées dans le run. Un checkpoint COCO
`person` ne peut pas être présenté comme un détecteur de joueurs. Les détecteurs
joueur seuls et leurs détecteurs arbitres séparés conservent leurs rôles dans le
tracking. Le conteneur d'inférence minimal utilise des poids locaux.

## Contrat gold pour tracking, identité et positions

Un fichier JSON gold utilise :

```json
{
  "schema_version": 1,
  "status": "draft",
  "reviewer": "à renseigner après revue",
  "split": "val",
  "source_sha256": "empreinte exacte de la vidéo",
  "frames": [{
    "frame_index": 450,
    "timestamp_s": 15.0,
    "segment_id": 0,
    "persons": [{
      "person_id": "identifiant humain persistant",
      "bbox": [100, 200, 150, 350],
      "team_id": null,
      "number": null,
      "position_m": null,
      "expected_identity": "abstain"
    }]
  }]
}
```

Cet exemple décrit le format, pas une annotation exploitable. Le statut `reviewed`
ne doit être posé qu'après revue humaine. Toutes les personnes cibles visibles
de chaque frame sélectionnée doivent être annotées. Les boîtes sont en pixels
source, les positions au sol en mètres dans la même orientation que le segment.
Les numéros sont des chaînes : `0` et `00` restent différents. Les identifiants
humains ne sont pas ceux du tracker. Une position ou identité incertaine reste
`null`. Les timestamps et segments sont liés aux sorties du run inspecté.

```bash
.venv-players/bin/python -m mlops.gold \
  --gold /gold/revu.json --run /analyse/terminee --output /nouveau/rapport.json
```

Le rapport mesure IDF1 via affectation globale des identités compatibles en IoU,
changements d'ID et fragmentation sur frames gold consécutives, précision des
identités acceptées, couverture, erreurs médiane/P95 de projection, et erreur de
distance sur les mêmes intervalles consécutifs valides. Une identité attribuée à
une fausse détection est pénalisée. L'identité d'une personne dont la vérité est
inconnue est signalée comme invérifiable, avec un compteur des violations
d'abstention explicite. Aucun HOTA n'est calculé.

La formulation IDF1 s'appuie sur la logique d'affectation globale de
[TrackEval Identity](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py).
Les contrôles analytiques incluent une rupture d'ID connue, absence de prédiction,
confusion `0`/`00`, faux sujet nominatif et couverture des intervalles de distance.
Une référence manuelle n'établit pas une précision physique parfaite.

Pour `split: test`, geler avant la lecture des scores :

```bash
.venv-players/bin/python -m mlops.gold --gold /gold/test.json \
  --run /analyse/terminee --freeze-recipe /nouveau/gold.frozen.json
.venv-players/bin/python -m mlops.gold --gold /gold/test.json \
  --run /analyse/terminee --frozen-recipe /nouveau/gold.frozen.json \
  --output /nouveau/test-result.json
```

Le gel lie la vérité terrain, les artefacts analysés, le code et le seuil IoU.
Il n'atteste pas que personne n'a déjà consulté le test.

## Bilan automatique de chaque analyse

Les nouvelles analyses finalisées produisent `health.json`, référencé par son
empreinte dans `run.json`. Pour contrôler un ancien run :

```bash
.venv-players/bin/python -m mlops.health \
  --run /analyse/terminee --output /nouveau/health.json
```

Les champs indiquent les dénominateurs : frames calibrées/frames, localisations
ballon/frames, positions projetées/observations joueur, identités nominatives/
observations joueur. Les fragments sont comptés par segment et track. Ce sont
des disponibilités, sans mesure de précision ou de drift faute de vérité terrain.

## Démo et conteneurs

Le viewer contient seulement Python standard et les fichiers d'interface.
`examples/demo/` fournit une vidéo H.264 et des résultats scriptés sans données
NBA. Les chemins vidéo relatifs se résolvent sous `--video-root` et restent soumis
au contrôle de racine autorisée.

```bash
docker compose up --build -d viewer
# http://127.0.0.1:8766
docker compose down
```

L'image CPU d'inférence utilise `deploy/runtime-cpu.lock`, une image Python
épinglée par digest, un utilisateur nommé non-root et des caches sous `/tmp`.
Le bundle `make bundle-models` contient les poids locaux, leurs empreintes,
une configuration CPU et le dépôt TrackNet épinglé. La création refuse d'écraser
un bundle. Elle n'accorde pas de droit de redistribution sur les poids.

```bash
make bundle-models
make verify-model-bundle
make runtime-image
make infer-container VIDEO=/videos/clip.mp4 OUTPUT=runs/mon-run MAX_FRAMES=150
```

L'inférence utilise les volumes locaux et `--network none`. Le profil Docker
minimal ne contient ni serveur MLflow, ni TensorFlow, ni entraîneur RF-DETR.
La pose CPU est exécutée par ONNX Runtime CPU ; aucun GPU n'est requis.
