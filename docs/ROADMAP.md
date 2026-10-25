# Feuille de route — version portfolio MLOps junior

Établie le 25 octobre 2026 à partir de
[l'audit du projet](ETAT_DES_LIEUX_2026-10-25.md).

**Objectif de sortie :** un dépôt présentable pour candidater, avec une démo
réelle de 30 secondes, deux exemples commentés, une sélection de modèle
justifiée, un benchmark indépendant limité et une exécution reproductible.

Le README est restructuré. Les six entraînements ballon sont terminés. Les
outils d'évaluation, de registre, de conteneurisation et les tests existent.
La suite consiste à les réunir dans une version cohérente et vérifiée.

## Ordre à suivre

| Étape | Travail principal | Livrable de fin |
| --- | --- | --- |
| 0 | Préserver l'état actuel et identifier les versions | Point de reprise et inventaire des artefacts |
| 1 | Comparer les checkpoints ballon existants | Modèle retenu, rapport et configuration de démo |
| 2 | Terminer le parcours de démonstration | Run réel de 30 secondes et viewer lisible |
| 3 | Mesurer la recette figée sur de nouveaux matchs | Rapport indépendant et erreurs expliquées |
| 4 | Enregistrer et documenter les exemples | Vidéo principale, deux exemples et README complété |
| 5 | Vérifier et publier la release | Clone reproductible, CI verte et version identifiée |
| 6 | Préparer la présentation pour les candidatures | Texte CV et présentation technique courte |

La collecte et l'annotation des matchs test peuvent commencer dès l'étape 0.
Leurs scores restent fermés jusqu'au gel de la recette à la fin de l'étape 2.

## 0 — Préserver un point de départ

- [x] Conserver les modifications locales, y compris les fichiers non suivis et
  l'état partiellement stagé du viewer, dans un point de reprise local vérifié.
  Copie conservée sous `release/git-backup-2026-10-25/`, hors Git. Préparer ensuite
  des commits cohérents sans perdre ce travail.
- [x] Inventorier les checkpoints : fichier, SHA-256, run MLflow, époque et
  critère de sauvegarde. Distinguer `best.pt`, `best_f1.pt` et `last.pt`.
  [Inventaire des 19 fichiers](../reports/ball-comparison-2026-10-25/checkpoint-inventory.json).
- [x] Conserver les identifiants d'exports propres aux expériences existantes.
  Ne pas remplacer leurs données sous le même identifiant.
- [ ] Réserver deux nouveaux clips issus de matchs absents du développement pour
  le test. Vérifier leur provenance et noter les inconnues concernant les
  données d'entraînement des poids importés.

**Terminé quand :** le travail actuel est récupérable et chaque checkpoint à
comparer est associé à un fichier et à ses données d'origine.

**Répartition :** je prépare l'inventaire et le point de reprise ; tu fournis ou
confirmes les nouveaux clips test disponibles. Leur absence ne bloque pas
l'évaluation sur validation de l'étape suivante.

## 1 — Sélectionner le modèle ballon

- [x] Comparer les six campagnes complètes : YOLO26 S/M/L, TrackNet V3, V5 et
  V5+TOTNet, plus l'ancien V5 de deux époques. Évaluer les 12 checkpoints
  `best.pt` et `best_f1.pt` archivés, avec leur provenance MLflow.
- [x] Geler les exports et le protocole avant inférence, vérifier l'équivalence
  des pixels et annotations entre formats. Comparer les mêmes cibles causales
  avec une tolérance de 15 pixels source ; seuil heatmap 0,5, confiance YOLO 0,25.
  Garder `unknown_position: exclude`. Signaler les coupes non revues et
  l'absence de vrais négatifs et d'occultations localisables dans cette validation.
- [x] Rapporter précision, rappel, F1 de localisation, couverture, erreurs
  médiane/P95 et coût d'inférence sur le même matériel. Séparer coût du modèle
  et temps de la chaîne complète.
- [x] Examiner un échantillon des erreurs importantes et des scènes difficiles.
- [x] Choisir le candidat avec le meilleur compromis qualité/coût pour la démo,
  en justifiant les écarts et les limites. Aucun seuil de réussite n'est déduit
  des futurs résultats test. **YOLO26L `best_f1.pt`** est retenu pour la prochaine
  intégration : F1 0,9451 et latence médiane d'adaptateur 5,19 ms sur validation.
- [ ] Ajouter l'adaptateur ballon YOLO au pipeline vidéo et une recette chargeant
  ce checkpoint. Vérifier le comportement sur vidéo brute, les coupes et la
  compatibilité avec les sorties du viewer. Le comparateur seul ne clôt pas
  cette intégration ; la recette actuelle garde son V5 historique.

**Livrables :** protocole, inventaire, métriques par checkpoint, sélection,
compromis qualité/coût et limites dans
[`reports/ball-comparison-2026-10-25/`](../reports/ball-comparison-2026-10-25/README.md).
La configuration de démo doit ensuite charger le candidat compatible retenu.

**Terminé quand :** le même protocole permet de comparer les sept campagnes et
la configuration de démo charge le checkpoint retenu.

**Répartition :** je réalise les évaluations et le rapport ; tu confirmes la
revue humaine des annotations/coupes lorsque nécessaire.

**Périmètre :** E-BARD reste la référence personnes. Les adaptateurs YOLO/V3
du comparateur sont distincts du pipeline vidéo, qui prend en charge V5/V5+TOTNet.
Le choix d'un modèle YOLO/V3 pour la démo nécessiterait cette intégration et une
validation sur vidéo brute. Aucun nouvel entraînement n'est requis pour comparer
les campagnes existantes.

## 2 — Terminer le parcours réel de 30 secondes

- [ ] Inspecter quelques extraits de développement et retenir **30 secondes
  continues** avec jeu visible et projection exploitable. HOU/LAL est un candidat
  à examiner, pas un extrait déjà validé.
- [ ] Préparer un roster explicite si des noms sont affichés, puis générer un
  nouveau run avec la recette retenue. Utiliser un dossier de sortie neuf.
- [x] Ajouter durée mesurée et couverture à la fiche joueur. Expliquer que la
  couverture porte sur les intervalles suivis, pas sur toute sa présence réelle.
- [x] Remplacer « Terrain en temps réel » par « Terrain projeté ». Lecture,
  synchronisation, sélection et mesures indisponibles vérifiées dans un navigateur
  sur le cas synthétique.
- [ ] Vérifier ce même parcours sur le futur extrait réel de 30 secondes.
- [ ] Mesurer chargement, traitement et temps total sur cet extrait, avec le
  matériel et la configuration réellement utilisés.
- [ ] Figer modèles, seuils, configuration, export de référence et code exécuté
  avant de consulter les résultats test. Les empreintes complètent le commit
  lorsque l'état contient encore des modifications locales.

**Terminé quand :** le viewer lit les 30 secondes avec des sorties issues des
modèles, des statistiques inspectables et un manifeste complet. Une identité
incertaine reste inconnue ; aucune correction manuelle ne se fait passer pour
une prédiction du pipeline.

**Répartition :** je prépare les runs, les modifications du viewer et les
mesures ; nous revoyons l'extrait et les résultats visibles.

## 3 — Obtenir un premier benchmark indépendant

- [ ] Sur les deux matchs réservés, viser initialement un clip de 30 secondes
  par match et **30 à 50 images espacées par clip** pour les personnes. Ces
  nombres bornent le travail ; ils ne garantissent pas la représentativité.
- [ ] Revoir les positions ballon des plages évaluées et les coupes. Les cas
  inconnus restent inconnus, sans conversion automatique en négatifs.
- [ ] Annoter une séquence dense de **3 à 5 secondes par clip** avec des
  identités humaines persistantes pour mesurer le tracking. Ne pas recopier les
  IDs du tracker comme vérité terrain.
- [ ] Ajouter quelques repères terrain indépendants pour contrôler la
  projection. Ne publier une erreur de distance que si la référence disponible
  permet réellement de la calculer et d'en expliquer l'incertitude.
- [ ] Versionner les annotations revues, geler les recettes test et exécuter
  l'évaluation finale une fois la sélection terminée.
- [ ] Publier résultats par clip, supports, couverture et erreurs typiques,
  y compris lorsqu'ils sont décevants.

**Livrable prévu :** `reports/portfolio/independent_validation.md`, avec les
recettes, versions et rapports détaillés associés.

**Terminé quand :** un lecteur peut distinguer les scores de développement des
scores indépendants et comprendre sur quelles données ils reposent.

**Répartition :** tu assures la revue humaine ; je prépare les outils,
préannotations, exports et calculs. Une préannotation seule ne clôt pas la tâche.

Si les résultats test conduisent à modifier le modèle ou les règles, conserver
le premier rapport. Le jeu consulté devient une donnée de développement pour
ces modifications ; un nouveau test sera nécessaire pour une nouvelle mesure
indépendante. Le calendrier ne doit pas bloquer une première démo de
développement, présentée explicitement comme telle.

## 4 — Produire les médias et compléter le README

- [ ] Enregistrer la démo réelle de 30 secondes avec des calques lisibles.
  La capture utilise les sorties du run figé et conserve sa provenance.
- [ ] Produire deux exemples de 5 à 10 secondes : ballon difficile et
  coupe/calibration indisponible. Ajouter l'identification seulement si un cas
  vérifié apporte une explication utile.
- [ ] Pour chaque exemple, indiquer le comportement attendu, le résultat
  observé et une limite concrète.
- [ ] Compléter les emplacements du README : médias, checkpoint choisi,
  rapport comparatif, mesures de la chaîne et benchmark indépendant disponible.
- [ ] Garder le parcours synthétique Docker comme essai sans modèles.

**Terminé quand :** le README montre ce que produit le projet et conduit aux
preuves techniques. Les médias retenus sont adaptés à la diffusion envisagée.

**Répartition :** je prépare captures, intégration et texte ; tu valides la
présentation finale et le choix des médias à diffuser.

## 5 — Vérifier et publier une release reproductible

- [ ] Actualiser DVC pour les annotations retenues et vérifier que la version
  nécessaire est récupérable par un utilisateur autorisé. Conserver les anciens
  exports utilisés dans les rapports historiques.
- [ ] Finaliser les commits, choisir la licence source et vérifier les fichiers
  destinés au dépôt public.
- [ ] Relancer les contrôles adaptés aux modifications :

  ```bash
  make check-poc CHECK_OUTPUT=runs/checks/portfolio-release
  ```

- [ ] Depuis un clone propre, vérifier le viewer Docker, la lecture de la vidéo
  et le parcours documenté d'inférence avec les checkpoints explicitement
  installés. Réutiliser le run de 30 secondes pour comparer les sorties attendues.
- [ ] Publier les changements et obtenir une exécution GitHub Actions verte.
  Corriger les éventuelles différences entre environnement local et CI.
- [ ] Identifier la release par un tag et consigner versions, rapports et limites.

**Terminé quand :** le dépôt publié correspond à la démo, la CI est verte et
les instructions ont été exécutées depuis des sources propres.

**Répartition :** je gère les branches, commits, push, PR et vérifications CI.
Les messages expliquent les changements et leur validation, avec des titres
courts. La PR présente le résultat final ; elle est relue et corrigée avant fusion,
après réussite des vérifications CI.

## 6 — Préparer la présentation pour candidater

- [ ] Écrire une description CV en deux lignes, avec un résultat mesuré et son
  contexte : qualité, coût d'inférence ou réduction de temps de traitement.
- [ ] Préparer une présentation orale de 90 secondes : problème, architecture,
  choix de modèle, résultat, difficulté rencontrée et limite restante.
- [ ] Préparer une explication détaillée de quatre sujets : séparation des
  données, sélection du modèle, reproductibilité, promotion et rollback.
- [ ] Vérifier que le lien utilisé dans le CV mène directement au README et à
  la démo. Adapter le vocabulaire aux postes MLOps, ML Engineer ou Computer Vision.

**Terminé quand :** chaque affirmation du CV est reliée à un résultat ou à une
fonctionnalité vérifiable dans le dépôt.

## Règle pour rester dans le périmètre

Un travail entre dans cette version s'il améliore directement la démo, la qualité
mesurée, la reproduction ou la présentation. Reporter les autres extensions :
nouveaux modèles, statistiques de possession/tirs, service d'upload, orchestration
cloud et infrastructure supplémentaire.

**Prochaine action : intégrer YOLO26L au pipeline vidéo à l'étape 1, puis valider
le candidat sur l'extrait réel de 30 secondes de l'étape 2.**
