# Couverture des identités SAC–SAS

Une démo assistée de cinq secondes, avec dix joueurs nommés sur les 125 images,
est maintenant disponible : [résultat, portée et reproduction](DEMO.md).
Les observations du match complet restent conservées.

Run analysé : `runs/analysis/sac-sas-2026-10-27-native-cuda-final`.
Cet audit répond au manque de joueurs identifiés dans le viewer. Les résultats
affichés n'ont pas été modifiés par cet audit. Les calculs de synthèse ont été
effectués sur CUDA sous `CudaOnly` ; les fichiers JSON servent de preuves.

## Constats

- 155 824 observations nommées sur 583 507 détections de joueurs : disponibilité
  de 26,70 %, selon `health.json`. Ce chiffre ne mesure pas leur exactitude.
- 7 040 fragments de trajectoire. La médiane contient 29 observations ; un quart
  en contient au plus cinq. Les fragments ne sont pas des joueurs distincts.
- 5 144 fragments sans majorité numérique ; 2 610 sans aucun vote de couleur.
- 12 747 lectures acceptées sur 123 981 lectures OCR enregistrées.
- 12 453 épisodes sur 16 434 sans lecture OCR acceptée. Le champ historique
  `number_attempts` compte ici les lectures acceptées, pas toutes les tentatives.
- 858 épisodes possèdent une majorité numérique mais pas de majorité d'équipe.
- `Evidence.update` arrête les nouvelles lectures après une première majorité
  de trois votes, potentiellement obtenus sur cinq images consécutives. Une
  majorité numérique peut donc arrêter l'acquisition avant qu'une identité
  soit résoluble avec les preuves d'équipe disponibles.

Un assouplissement strict déjà utilisé par le backend historique (numéro unique,
trois lectures ≥ 0,95, majorité ≥ 90 %, couleur concordante et aucune preuve
adverse) ne débloquerait ici que 19 épisodes, soit au plus 1 654 observations
avant contrôle des collisions. Il ne suffirait pas à résoudre le manque global
d'identités. Voir `sparse-color-candidates.json` ; aucune de ces associations
n'a été appliquée.

## Améliorations à privilégier, sans nouveaux modèles

1. Nouvelle passe PARSeq ciblée sur les trajectoires anonymes, sur des images
   nettes réparties dans leur durée. Réutiliser bbox, poses et masques enregistrés,
   et conserver une seule contribution par image. Revérifier les majorités plutôt
   que verrouiller une identité après les premières lectures rapprochées.
2. Améliorer les preuves d'équipe sur les torses segmentés : le classifieur
   actuel exige une forte proportion de pixels sombres ou blancs, et laisse
   les tissus gris, ombragés ou mélangés à la peau sans classe. Vérifier les
   nouvelles règles sur des échantillons annotés avant propagation des noms.
3. Réduire la fragmentation et exploiter les continuités courtes non ambiguës,
   avec contrôle des croisements et doublons. La compensation du mouvement de
   caméra est absente de ce ByteTrack CUDA ; une évolution doit rester CUDA et
   conserver les poids des six modèles.

Le retraitement doit produire un nouveau run et recalculer ensemble les identités,
les distances et la santé. La comparaison doit mesurer la disponibilité et
contrôler visuellement un échantillon d'identités ; une hausse du nombre de noms
ne prouve pas une hausse de leur exactitude.
