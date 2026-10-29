# Démo SAC–SAS : identités sur cinq secondes

Extrait source **01:39,04 → 01:44,04**, images 2476 à 2600 incluses.
Vidéo 1920 × 1080, 25 images/s, 125 images. Le nouveau run est
`runs/analysis/sac-sas-demo-identites-verifiees-5s`.

La vidéo annotée est dans
`datas/test/SAC_SAS/demo/SAC_SAS-demo-assistee-verifiee-5s.mp4`.
La page locale est http://127.0.0.1:8767/ et le viewer est
http://127.0.0.1:8765/?run=sac-sas-demo-identites-verifiees-5s.

## Résultat et portée

Les 125 images contiennent chacune dix joueurs nommés, sans collision d'identité.
Sur les mêmes détections conservées, le run précédent nommait 777 observations
sur 1250 (62,16 %), contre 1250 sur 1250 dans l'extrait revu.
Ce résultat décrit **un extrait assisté**, et non une mesure d'identification
automatique sur le match complet. La couverture seule ne mesure pas la justesse.

Les dix identités initiales ont été contrôlées avec les numéros visibles et le
cinq officiel. Le play-by-play ESPN archivé donne le premier remplacement à
6:11 du premier quart ; le passage contrôlé va de 9:26 à 9:21, avant ce changement.
Les cas déduits du cinq sont signalés dans `demo-reviewed-seeds.json`.

Cinq décisions de détection sont explicitement revues dans ce même fichier :
un arbitre classé joueur, deux doublons partiels, et deux détections de vrais
joueurs conservées pendant une occultation malgré leur faible confiance.
Leurs coordonnées et scores du modèle restent inchangés. Aucune bbox n'est
inventée et aucun score n'est augmenté. Les exclusions de mesure des distances
restent appliquées aux observations peu fiables.

## Changements réutilisables

- `pipeline/cuda/recovery.py` relit les détections, poses et masques enregistrés
  avec NVDEC et PARSeq sur CUDA. Une seule contribution OCR est conservée par
  joueur et image, même avec plusieurs crops. La passe du contexte de 832 images
  a traité 19 842 crops en 52 secondes, avec les poids PARSeq d'origine.
- Les nouveaux crops fondés sur la bbox restent disponibles quand la pose est
  valide mais cadre trop étroitement le numéro. Le cas de Vassell coupait le « 2 »
  de « 24 ». Les seuils de qualité et le contrôle des occultations restent actifs.
- `Evidence` revérifie périodiquement un numéro confirmé, au lieu d'arrêter
  définitivement l'OCR après la première majorité rapprochée.
- `pipeline/cuda/lineup.py` fournit une affectation CUDA avec Kalman, géométrie,
  preuves d'équipe et identités initialisées. Son initialisation automatique
  refuse les égalités ambiguës. Ici, les ancres sont explicitement revues.
- Les doublons de corps sont filtrés par masques, boîtes et poses. Les bascules
  isolées de rôle sont lissées seulement avec un arbitre correspondant avant et
  après. Le seuil de confiance s'applique aux détections nommées et anonymes.
- Le rejeu efface les anciens conflits et raccords avant de recalculer les noms
  et les statistiques, pour éviter de conserver des annotations périmées.

Les six modèles et leurs poids sont conservés. Les détections, poses et masques
du run source sont réutilisés. Le run complet et ses observations sont intacts.

## Contrôles

- 160 tests pipeline passent : `refinement-all-tests.log`.
- Santé, horodatages et empreintes validés sur les 125 images dans le nouveau run.
- Comparaison NVDEC aux images source voisines : début aligné sur 2476 et fin
  sur 2600, avec l'image attendue comme meilleur appariement ; voir
  `demo-media-check.json`.
- Vérification du viewer sur les 125 images : dix identités distinctes par image,
  sans erreur JavaScript ; voir `demo-browser-check.json`.
- `demo-validation.json` conserve les vérifications et la continuité géométrique.
- Aucun calcul de tenseur CPU ni repli CPU dans la récupération, le suivi, les
  statistiques, le rendu et l'export. Les fichiers et métadonnées passent par le
  CPU. Le rendu utilise une police bitmap constante, des opérations CUDA et
  une conversion NV12 CUDA ; l'encodage est NVENC.

La vidéo annotée met en avant les joueurs, leurs noms et leurs masques. Pour
reproduire cette vue dans Match Analyse : joueurs en Bbox + Segmentation avec
noms, arbitres masqués, ballon et terrain décochés. Les métriques de projection
conservent les limites de calibration du run source ; cet extrait valide la
continuité des identités pour la démo.

## Reproduire

Utiliser des dossiers de sortie neufs :

```bash
.venv-players/bin/python -m pipeline.cuda.recovery \
  --run runs/analysis/sac-sas-2026-10-27-native-cuda-final \
  --output runs/recovery/nouvelle-passe-sac-sas --start 2476 --stop 3308 --step 2

.venv-players/bin/python -m pipeline.cuda.demo \
  --source runs/analysis/sac-sas-2026-10-27-native-cuda-final \
  --recovery runs/recovery/nouvelle-passe-sac-sas \
  --review reports/sac-sas-identity-audit-2026-10-28/demo-reviewed-seeds.json \
  --video datas/test/SAC_SAS/demo/SAC_SAS-identites-5s.mp4 \
  --output runs/analysis/nouvelle-demo-sac-sas

.venv-players/bin/python scripts/export_demo_gpu.py \
  --run runs/analysis/nouvelle-demo-sac-sas \
  --output datas/test/SAC_SAS/demo/nouvelle-demo-sac-sas.mp4
```
