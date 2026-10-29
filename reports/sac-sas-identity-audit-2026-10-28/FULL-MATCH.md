# Seconde passe d’identification SAC–SAS

Lancement du 28 octobre 2026 :
`runs/analysis/sac-sas-2026-10-28-identites-v2`.
Suivi : http://127.0.0.1:8766/.
Journal : `runs/sac-sas-2026-10-28-identites-v2.log`.
Le résultat source reste dans `sac-sas-2026-10-27-native-cuda-final`.

## Traitement

- Réutilisation des six modèles et des détections, poses, masques, ballon et
  calibrations enregistrés. Vérification des empreintes des fichiers source
  et des poids avant réutilisation.
- Parcours NVDEC des 59 827 images, relecture PARSeq toutes les deux images
  avec six cadrages possibles, sans arrêt après confirmation. Un seul vote
  par joueur et image. Une lecture hors effectif ne remplace pas une lecture
  autorisée plus ancienne.
- Preuves supplémentaires de couleur pour les maillots sombres et blancs :
  deux régions de torse segmentées doivent s’accorder. Les couleurs source
  déjà connues restent prioritaires ; gris intermédiaire et peau dominants
  restent indéterminés.
- Suppression des doubles représentations d’un corps selon masques, poses
  et boîtes. Conservation des détections faibles ; aucun relèvement de score.
  Correction des bascules isolées joueur/arbitre avec un arbitre correspondant
  avant et après.
- Raccords des courtes interruptions d’une trajectoire avec géométrie sûre,
  couleur compatible et absence de voisin ambigu. Coupure aux changements
  soutenus de numéro. Récupération prudente des numéros uniques avec peu de
  votes de couleur.
- Propagation avant/arrière par appariement CUDA univoque des corps et
  estimation robuste de la translation commune de caméra. Arrêt aux coupes,
  couleurs opposées, ambiguïtés, ancres contradictoires et à quatre secondes
  de la dernière ancre. Les numéros sont résolus de nouveau, sans réutiliser
  les anciennes suppressions de conflits.
- Recalcul complet des statistiques puis vérification des 59 827 indices,
  horodatages de présentation, identités et identifiants uniques, et empreintes.
  Le suivi englobe toutes ces étapes ; 100 % exige leur succès.

Les calculs de tenseurs, l’OCR, le décodage vidéo et les statistiques sont sur
CUDA. Contrôle GPU strict, arrêt sur violation, aucun repli CPU. Le CPU assure
le contrôle, les métadonnées, la lecture et l’écriture des fichiers.

## Compositions officielles : portée précise

Les compositions successives du premier quart sont reconstituées depuis les
remplacements officiels archivés. Elles peuvent compléter un cinquième joueur
uniquement si quatre coéquipiers sont déjà identifiés, les cinq sont visibles,
et toutes les compositions compatibles donnent la même identité manquante.

Cette contrainte est limitée aux images 2476–2600, seul intervalle dont la
synchronisation avec le premier quart a été vérifiée. Aucun alignement
automatique du chronomètre sur le match complet n’est disponible. Les pauses,
ralentis et changements pendant les interquart-temps empêchent de déduire
les compositions du seul temps vidéo. Aucune composition n’est imposée ailleurs.
Les dix ancres manuelles et les cinq décisions de détection de la démo ne sont
pas appliquées au nouveau résultat.

## Validation et interprétation

167 tests pipeline passent (`runs/review/sac-sas-2026-10-28/full-refinement-all-tests.log`).
Un essai de bout en bout sur les 100 premières images, avec les mêmes détections
retenues, passe de 170/756 noms (22,49 %) à 203/756 (26,85 %). Horodatages,
identifiants, statistiques et empreintes contrôlés ; aucune opération de tenseur
CPU ni repli. Cet essai ne préjuge pas du gain sur le match complet.

Après succès, `identity-quality.json` compare la couverture initiale et finale
sur les détections conservées et publie aussi les effectifs initiaux, les
doublons retirés et les preuves supplémentaires. `health.json` valide
l’intégrité. La couverture n’est pas une mesure de justesse des noms et aucune
promesse de 100 % d’identification automatique n’est faite.

## Reproduction

Utiliser une sortie neuve :

```bash
.venv-players/bin/python -m pipeline.cuda.refine \
  --source runs/analysis/sac-sas-2026-10-27-native-cuda-final \
  --output runs/analysis/nouvelle-identification-sac-sas \
  --step 2 \
  --clock-review reports/sac-sas-identity-audit-2026-10-28/demo-reviewed-seeds.json
```

Une seule seconde passe peut s’exécuter à la fois (verrou de processus).
En cas d’échec, `run.json`, `progress.json`, `failure.txt` et les fichiers
intermédiaires sont conservés. Une récupération OCR terminée peut être réutilisée
avec `--recovery chemin/vers/recovery`, après contrôle de sa provenance et de
son intervalle. Le traitement ne publie pas de résultat déclaré terminé si un
contrôle échoue.
