# Plan proposé — pipeline de détection des joueurs

Date : 22 septembre 2026. Statut : **validé par l'utilisateur ; phases 1 à 7 implémentées**.

La phase 1 a livré les contrats joueurs, le socle commun et les tests de
non-régression. Détail et commandes : [training/players/README.md](../training/players/README.md).
La phase 2 ajoute l'éditeur de boîtes et la préannotation YOLO/cache NBA avec revue
explicite. L'import de datasets externes a été retiré à la demande de l'utilisateur
le 23 septembre 2026. La phase 3 ajoute les exports YOLO/COCO reproductibles,
l'audit des splits et la publication atomique d'une paire vérifiée.
La phase 4 ajoute les adaptateurs YOLO26/RF-DETR, l'environnement dédié et épinglé,
les recettes, les checkpoints complets, la reprise stricte, le chargement sans
entraînement et le suivi MLflow explicite. Voir le
[guide d'entraînement](../training/players/TRAINING.md).
La phase 5 ajoute l'évaluation COCO commune, les rapports et comparaisons à
protocole identique, le benchmark d'inférence et le suivi `players-evaluation`.
Voir le [guide d'évaluation](../training/players/EVALUATION.md) pour les commandes,
les essais courts et les limites (occultation par groupes d'images, vitesse vidéo
de bout en bout non mesurée dans l'évaluateur d'images).
La phase 6 ajoute l'orchestration locale avec reçus vérifiés et progression
atomique, l'inférence vidéo sans annotations et le registre `players-detector`
avec promotions explicites et réversibles par alias. Voir le
[guide du parcours local](../training/players/PIPELINE.md).
La phase 7 ajoute la CI CPU offline, les smokes GPU activables explicitement,
les contrats d'intégrité complémentaires et une démonstration synthétique avec
registre local. Voir [la validation finale](../training/players/VALIDATION.md) et
[son bilan](../training/players/VALIDATION_RESULTS.md). L'exécution technique ne
constitue pas une validation de qualité sur les matchs ; l'annotation humaine
et la recette finale sur données réelles restent à réaliser.

Construire dans `training/players/` une chaîne reproductible allant des annotations
à la comparaison des modèles et à leur utilisation sur une vidéo. Reprendre les
garanties du pipeline balle, en adaptant les données et les métriques à la détection
de plusieurs personnes par image.

## 1. Ce que l'inspection a établi

| Sujet | Vyneros | Enseignement pour les joueurs |
| --- | --- | --- |
| Annotation | Annotateur balle local avec sidecar JSON et sauvegarde protégée | Créer le contrat de boîtes multiples et une interface de correction |
| Données | `datas/train`, `datas/val`, métadonnées match/vidéo, DVC et configuration R2 | Réutiliser les vidéos, leurs identités et leurs splits |
| Exports | YOLO, COCO, formats temporels ; manifestes, empreintes, contrôle d'intégrité, publication avec restauration en cas d'erreur | Conserver YOLO et COCO, sans formats temporels |
| Entraînement | YOLO, TrackNet V3/V4/V5 et variante TOTNet ; configurations validées, checkpoints et reprise | Reprendre les contrats généraux, avec des adaptateurs de détecteurs |
| MLflow | Runs explicites, provenance, paramètres, métriques, checkpoints ; URI configurable | Même serveur/base possible, expériences joueurs distinctes |
| Évaluation | Évaluateur commun aux deux variantes V5 ; métriques, revue CSV et vidéos | Ajouter un évaluateur de boîtes commun aux familles de détecteurs |
| Avancement | Le run quick ouvert dans l'IDE possède un `progress.json` détaillé ; aucun producteur générique de ce fichier trouvé dans `training/` ou `scripts/` | Intégrer explicitement ce suivi aux commandes joueurs |
| Module joueurs | `training/players/` existe mais est vide | Implémentation à construire |

Dans `../NBA`, les éléments utiles sont le convertisseur SportsMOT MOT → YOLO,
le regroupement des clips par vidéo pour les splits, le trainer YOLO partagé et
le contrat d'inférence `bbox + confidence + class_id`. Les poids
`models/Joueurs_26L.pt` et `models/Joueurs_26X.pt` sont présents. En revanche,
le dataset `data/sportsmot_yolo/` référencé par sa recette n'est pas présent, et
ses documents indiquent qu'il manque une évaluation sur les vidéos de clubs.
RF-DETR y est mentionné comme alternative, sans adaptateur joueurs trouvé.

Dans `Vyneros/datas/`, l'inspection a trouvé des caches de détection joueurs,
des annotations balle/terrain et des données de maillots, mais pas de vérité
terrain dédiée aux boîtes des joueurs. **Un cache de prédictions n'est pas une
annotation validée.** Le pipeline pourra être validé techniquement avec des
fixtures ; une mesure réelle de qualité demandera des images annotées.

Les modifications locales déjà présentes dans le dépôt seront conservées.
Le projet NBA sert de référence en lecture ; le nouveau module devra fonctionner
sans import ni chemin obligatoire vers `../NBA`.

## 2. Périmètre proposé

- Une classe : `0 = player`, pour les joueurs participant au jeu. Règles explicites
  pour les occultations, troncatures et joueurs sortant momentanément du terrain.
- Arbitres, public, entraîneurs et banc : hors classe cible ; exemples utiles pour
  mesurer les faux positifs. Les situations ambiguës ne deviennent pas des négatifs
  par défaut.
- Pas de numéros de maillot, identification nominative, attribution d'équipe,
  tracking, pose ou analyse des tirs dans cette livraison.
- Les premières données sont les vidéos présentes dans Vyneros. Le protocole
  permettra ensuite un benchmark de clubs par salle, sans supposer que ce jeu
  annoté existe déjà.
- Deux familles entraînables et évaluables : **YOLO26 et RF-DETR**. Plusieurs tailles
  YOLO pourront être comparées par recette ; aucun modèle ne sera désigné meilleur
  avant mesure. TrackNet reste propre à la balle.

## 3. Lot 1 — contrats et socle partagé

Créer `training/players/{annotator,export,learning,evaluation,configs,tests}`,
un Makefile délégué et une documentation de démarrage.

Définir un schéma versionné d'annotations, de prédictions et de manifestes :
vidéo/image d'origine, index de frame, coordonnées source, classe, état de revue,
identité du match, salle éventuelle et provenance.

Extraire dans `training/common/` uniquement les utilitaires réellement communs :
lecture/validation de configuration, provenance, suivi MLflow, écritures atomiques
et identités des sources. Le logger actuel contient des chemins spécifiques à
`training/ball` : les rendre paramétrables. Préserver les entrées publiques balle
et ses formats/checkpoints. Les boucles TrackNet et les métriques balle restent
dans leur module.

**Validation :** contrats joueurs testés et tests balle de non-régression verts.

## 4. Lot 2 — annotation et constitution du dataset

Créer un sidecar proposé `video.mp4.playersann.json` avec plusieurs boîtes par
frame, dans les coordonnées de la vidéo source. Distinguer au minimum les frames
non annotées, proposées automatiquement et entièrement vérifiées, y compris les
frames vérifiées sans joueur.

Ajouter un annotateur local : création, déplacement, redimensionnement et
suppression des boîtes, navigation entre frames sélectionnées, validation de la
frame complète, sauvegarde atomique et détection des modifications concurrentes.

Prévoir deux sources de préannotations : un checkpoint de détecteur et les caches
NBA compatibles. Vérifier la correspondance vidéo/frame et conserver la provenance.
Les propositions restent à corriger ; elles ne sont pas exportées comme vérité
terrain tant qu'elles ne sont pas validées.

**Validation :** corrections persistées sans perte, plusieurs joueurs par frame,
boîtes aux bords, négatifs explicites et exclusion des annotations incomplètes.

## 5. Lot 3 — exports reproductibles et splits

Produire `exports/players/yolo/` et `exports/players/coco/` depuis les mêmes frames
validées. Conserver `dataset_id` commun et `export_id` propre à chaque format,
inventaire avec empreintes, configuration résolue, statistiques et transformations
de coordonnées réversibles. Reprendre les protections de publication des exports
balle et empêcher l'écrasement d'un répertoire étranger.

Le split reste celui de `datas/train|val|test`. Vérifier les collisions de vidéo,
de contenu et de match entre splits ; si le protocole annonce des salles inédites,
exiger `venue_id` et vérifier aussi la séparation par salle. Ne pas redistribuer
silencieusement les données existantes. Le test est exclu de l'entraînement et
du réglage des seuils.

Tracer les annotations sources avec le DVC existant. Les exports restent
régénérables. Les manifestes et rapports conservent la provenance des sources locales.

**Validation :** mêmes identités d'images et mêmes boîtes dans les deux exports,
empreintes cohérentes, absence de fuite et échec explicite sur données invalides.

## 6. Lot 4 — entraînement multi-modèles et MLflow

Implémenter d'abord YOLO26 en adaptant les garanties du trainer balle, puis RF-DETR
derrière le même contrat d'exécution. RF-DETR dispose d'un entraînement sur données
personnalisées et de mécanismes de reprise ; la version et les hooks seront épinglés
après vérification de compatibilité avec les dépendances locales.
Références : [entraînement](https://rfdetr.roboflow.com/latest/learn/train/),
[personnalisation](https://rfdetr.roboflow.com/latest/learn/train/customization/).

Prévoir des recettes smoke, quick et complètes, avec budget, seed, résolution,
augmentations, initialisation, précision et matériel explicites. Distinguer une
reprise stricte d'un fine-tuning : restaurer les états nécessaires du framework
pour la première et créer une nouvelle expérience pour le second. Documenter
les limites de reproductibilité constatées pour chaque adaptateur.

Chaque exécution crée un run explicite dans `players-training`, avec paramètres,
empreintes du code et des données, environnement, poids initiaux, pertes,
métriques par epoch, temps, checkpoints complet et meilleur. Désactiver les
intégrations automatiques qui créeraient des runs en double. Relier toute reprise
à son run parent. Préserver les artefacts exploitables et le statut sur interruption.

Accepter `MLFLOW_TRACKING_URI` comme aujourd'hui, avec base locale par défaut.
Les anciens poids NBA sont des références importées facultatives, avec SHA-256
et provenance déclarée ; ils ne deviennent pas automatiquement les poids initiaux.

**Validation :** entraînement court réel pour chaque famille, métriques MLflow,
rechargement des checkpoints et reprise vérifiée après interruption.

## 7. Lot 5 — évaluation commune et comparaison

Normaliser toutes les prédictions en boîtes source, scores et classe. Utiliser
un évaluateur COCO commun pour AP@50 et AP@50:95, et un appariement explicite
pour précision/rappel/F1 aux seuils choisis. Les métriques natives des frameworks
restent utiles au suivi mais ne constituent pas le classement commun.

Produire les résultats globaux et par match, clip, salle renseignée, taille des
joueurs et occultation annotée. Documenter le protocole : IoU, scores minimum,
limite de détections, pré/post-traitement et périmètre des annotations. Ne pas
tronquer la courbe AP avec le seuil d'affichage vidéo. Les groupes sans support
restent signalés et ne reçoivent pas de performance fictive.

Mesurer aussi latence médiane/P95, débit et mémoire GPU, sur matériel et précision
identiques, avec préchauffage et synchronisation. Distinguer temps modèle et temps
du traitement vidéo. Les différentes résolutions restent visibles dans le rapport.

Créer `players-evaluation`, lié au run entraîné ou à la référence importée, puis
un rapport de comparaison CSV/JSON/HTML : qualité, vitesse, mémoire et conditions
expérimentales. Comparer uniquement des runs portant sur le même jeu d'évaluation
et le même protocole ; identifier les différences de budget, préentraînement et
résolution. Exclure les smoke tests du classement par défaut. Réserver les essais
multi-seeds aux finalistes et conserver le test pour une recette figée.

Artefacts : `predictions.jsonl`, `metrics.json`, courbes, `review.csv`, images et
vidéos avec vérité terrain/prédictions, repérage des FP/FN et indices source.

**Validation :** cas connus de TP/FP/FN, doublons, images vides, coordonnées et
comparaison de prédictions identiques venant des deux adaptateurs.

## 8. Lot 6 — exécution démontrable et registre

Ajouter un orchestrateur local qui enchaîne validation des données, export,
entraînement, évaluation et comparaison à partir d'un YAML. Traiter les essais
GPU séquentiellement par défaut. Écrire un `progress.json` atomique avec étape,
run MLflow, epochs/batches, temps écoulé, estimation lorsqu'elle est disponible,
sorties et erreurs. Relancer depuis les étapes terminées uniquement si leurs
empreintes correspondent toujours à la configuration.

Prévoir les commandes suivantes, avec variantes de recettes :

```text
make annotate-players VIDEO=...
make preannotate-players VIDEO=... CHECKPOINT=...
make export-players
make train-players-yolo CONFIG=... [RESUME=...]
make train-players-rfdetr CONFIG=... [RESUME=...]
make evaluate-players CHECKPOINT=...
make compare-players CONFIG=...
make pipeline-players CONFIG=...
make predict-players VIDEO=... CHECKPOINT=...
make register-players CONFIG=...
make test-players
```

L'inférence vidéo fonctionne sans annotations, écrit les boîtes et une vidéo de
contrôle, et traite les frames avec une mémoire bornée. Elle partage le même
prétraitement que l'évaluation.

Enregistrer les candidats dans un modèle MLflow `players-detector`, avec lien
vers données, entraînement et évaluation. Fournir une promotion explicite par
alias, reproductible et réversible ; aucune promotion automatique basée sur un
smoke test. Les alias sont le mécanisme prévu dans les
[workflows officiels du registre MLflow](https://www.mlflow.org/docs/latest/ml/model-registry/workflow/).

Le service web, TensorRT, Dagster et le monitoring en production restent des
extensions ultérieures. La livraison couvre l'orchestration locale, le registre
et une démo vidéo reproductible ; leur existence ne sera pas confondue avec un
déploiement en production.

**Validation :** parcours complet sur un petit dataset, vidéo produite, provenance
navigable dans MLflow, enregistrement/rechargement d'un candidat et reprise du
parcours sans duplication injustifiée.

## 9. Lot 7 — validation finale et documentation

Ajouter les tests de contrats et d'intégration nécessaires, une CI CPU sur données
synthétiques, et des smoke tests GPU explicitement activables pour les deux
familles. Exécuter la non-régression balle après l'extraction des composants
communs et à la livraison.

Documenter l'installation, les conventions d'annotation, les commandes, les
artefacts, la connexion MLflow, la restauration DVC et une démonstration de bout
en bout. Donner les résultats mesurés, le volume réellement annoté et les limites.
La réussite technique du pipeline et la qualité du détecteur sont deux critères
distincts : pas de seuil de performance finale inventé avant la première baseline.

## 10. Périmètre validé

Le périmètre approuvé comprend : une classe `player`, annotation et
préannotation locales, exports YOLO/COCO, deux familles YOLO26 + RF-DETR,
évaluation commune, comparaison, MLflow, registre et parcours vidéo avec suivi
d'avancement. L'ordre d'implémentation est celui des lots ci-dessus.

L'annotation humaine du dataset et les entraînements longs constituent des étapes
distinctes des tests techniques. Le plan ne suppose ni annotations déjà validées,
ni performances acquises, ni budget GPU illimité.

**Plan validé. L'implémentation avance phase par phase, sur demande de l'utilisateur.**
