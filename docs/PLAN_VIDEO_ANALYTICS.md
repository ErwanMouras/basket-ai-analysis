# Plan proposé — pipeline vidéo NBA et distance parcourue

Date : 27 septembre 2026. Statut : **validé par l'utilisateur ; implémentation
POC livrée le 28 septembre 2026**, voir [utilisation](../pipeline/README.md) et
[validation](../pipeline/VALIDATION.md). Le texte ci-dessous conserve le cadrage
validé. La baseline E-BARD apporte désormais une détection joueurs/arbitres ;
la précision réelle des distances reste à évaluer sur des annotations dédiées.

## 1. Objectif et périmètre

Construire une entrée unique : **vidéo + effectif JSON facultatif → observations
unifiées → distance observable par joueur → JSON de statistiques**.

Le pipeline doit intégrer joueurs, arbitres, balle, terrain, pose et numéros de
maillot. Les modèles restent remplaçables par configuration. Aucun choix de
modèle définitif, entraînement systématique, configuration DVC, service web ou
outil de visualisation ne fait partie de cette livraison.

Proposition de référence pour la V1 : traitement local, hors ligne, d'extraits
de jeu continus. Les vidéos avec coupes sont segmentées ; leur couverture est
explicitée. Un match télévisé peut être traité techniquement, mais son résultat
ne devient pas une distance totale de match : replays, ralentis, absences du
champ et changements d'identité exigent un traitement spécifique.

Prévoir des plages temporelles d'inclusion/exclusion dans la configuration pour
écarter les replays ou séquences non pertinentes connus. Leur reconnaissance
automatique et la synchronisation avec l'horloge du match sont hors V1 proposée.
Sans ces plages, la métrique porte sur le contenu vidéo, susceptible de montrer
plusieurs fois une même action. Ce périmètre reste à confirmer par l'utilisateur.

## 2. État réel des briques inspectées

| Brique | Disponible | Travail nécessaire |
| --- | --- | --- |
| Joueurs | Adaptateurs YOLO/RF-DETR, boîtes en pixels source | Adapter au contrat d'analyse ; distinguer personne générique et joueur |
| Arbitres | Aucune classe ni chaîne dédiée trouvée dans `training/` | Ajouter un adaptateur de rôles/multiclasse et une baseline réellement capable de distinguer joueurs/arbitres |
| Suivi | BoT-SORT/ByteTrack, IDs locaux, remises à zéro | Réutiliser, séparer rôle et identité, synchroniser les segments |
| Pose | RTMPose COCO17 par boîte, lots | Réutiliser via adaptateur ; rattacher par identifiant d'observation |
| Maillot | PARSeq, sélection des crops, vote temporel, résolution d'effectif | Rendre l'effectif facultatif ; séparer inférence OCR et fusion temporelle |
| Terrain | Repères, homographie, propagation caméra, coupes, positions en mètres | Séparer inférence et état temporel pour permettre l'ordonnancement |
| Balle | Évaluation TrackNet V5/V5+TOTNet, fenêtres de trois images | Créer une entrée vidéo indépendante du dataset d'évaluation et restituer les pixels source |
| Statistiques | Pas de moteur de distance identifié | Créer trajectoires filtrées, règles de validité, agrégation et export |

Points vérifiés dans le code :

- `training/players/predict.py` enchaîne actuellement détection, calibration,
  tracking, pose, OCR, projection et rendu de façon séquentielle.
- L'effectif est chargé obligatoirement depuis le `jersey.json` voisin de la
  vidéo quand l'OCR est activé. `JerseyRecognizer` dépend directement du roster.
- `training/players/contracts.py`, le tracker et la pose attendent une classe
  `0 = player`. Le checkpoint générique local de personnes ne permet pas à lui
  seul de garantir la distinction joueur/arbitre/public.
- Le prédicteur balle vérifie actuellement l'identité du dataset d'évaluation.
  La nouvelle entrée vidéo doit partager le chargement/prétraitement utile sans
  supprimer les protections de l'évaluateur existant.
- Les références balle peuvent modifier l'environnement d'import Python : un
  worker isolé peut être nécessaire pour éviter les conflits entre modèles.
- La calibration utilise les détections de personnes pour masquer le mouvement
  du parquet. L'inférence des repères est indépendante, la mise à jour complète
  de la calibration ne l'est pas.
- Tracking et calibration exigent actuellement des indices consécutifs ; les
  segments et les indices vidéo doivent être harmonisés dans les adaptateurs.
- Le suivi ne fait pas de réidentification d'une personne après une coupe.
- Les validations terrain existantes mesurent une couverture, pas une erreur
  métrique des trajectoires. Elles documentent aussi un écart d'une image entre
  le nombre déclaré dans un conteneur et le nombre réellement décodé.

Références locales : [parcours vidéo](../training/players/predict.py),
[tracking](../training/players/tracking.py),
[OCR](../training/jersey/temporal.py),
[calibration](../training/court/calibration.py),
[balle](../training/ball/evaluation/inference.py),
[validation terrain](../training/court/VALIDATION.md).

## 3. Entrées et comportement attendu

Interface envisagée, à créer :

```bash
python -m pipeline analyze --video match.mp4 --config configs/analysis.yaml --output runs/analysis/match
python -m pipeline analyze --video match.mp4 --roster jersey.json --config configs/analysis.yaml --output runs/analysis/match-roster
```

- Vidéo seule : IDs anonymes, OCR possible sans liste autorisée, noms et équipes
  inconnus. L'absence de roster ne désactive ni pose, ni OCR, ni distance.
- Avec `--roster` : validation du format existant, copie figée, filtrage des
  numéros éligibles et résolution des identités quand les preuves suffisent.
- Un roster explicitement fourni mais invalide provoque une erreur avant le
  chargement des modèles. Aucun chargement implicite du fichier voisin dans la
  nouvelle commande : le comportement dépend des arguments/configuration.
- Préserver les numéros comme chaînes, notamment `0` et `00`. Un numéro partagé
  par deux équipes reste ambigu. L'attribution visuelle d'équipe est hors V1.
- Poids locaux, adaptateur, device, cadence, résolution et limites de ressources
  configurables par brique. Pas de téléchargement implicite à l'inférence.
- Profil complet avec les six capacités ; profils réduits explicitement
  configurables pour diagnostic. Une capacité indisponible est signalée.
- Une personne de rôle inconnu n'entre pas dans les statistiques de joueurs.
  Absence de capacité arbitre et absence d'arbitre détecté sont deux états distincts.
- Précontrôle des capacités requises avant exécution. En cours de run, une panne
  OCR/balle/pose peut produire une sortie partielle documentée ; une panne de
  détection/suivi empêche de poursuivre la distance. Une calibration localement
  indisponible laisse les observations intactes et suspend les mesures métriques.

## 4. Architecture et parallélisme

Créer un package `pipeline/`, distinct des parcours d'entraînement. Réutiliser
les briques existantes par adaptateurs ; extraire uniquement les couplages qui
empêchent leur utilisation. Conserver les commandes historiques fonctionnelles.

```text
Décodage unique : image, index, timestamp source
  ├─ Détection personnes / rôles ───────────┐
  ├─ Détection des repères terrain ────────┼─ Calibration et coupes ordonnées
  └─ Détection balle, fenêtres temporelles │
                                          ↓
Personnes + coupes ── Tracking ordonné ── Pose par lots ── Sélection crops OCR
                           │                    │                  ↓
                           └─ Calibration ──────┴─ Projection    Lectures OCR
                                                   │             ↓
                                                   │         Vote / identité
                                                   └──────┬──────┘
                                                          ↓
                                       Observations unifiées et trajectoires
                                                          ↓
                                               Distance et statistics.json
```

Le schéma décrit les dépendances, pas un worker obligatoire par boîte.

Principes d'exécution :

1. Décoder une fois ; partager les images via un tampon borné et libérer chaque
   image après son dernier consommateur. Ne pas conserver la vidéo en RAM.
2. Soumettre les inférences indépendantes en parallèle lorsque les ressources
   le permettent. Recouvrir décodage CPU, calcul GPU, OCR CPU et écritures.
3. Charger chaque modèle une seule fois par worker ; limiter les threads CPU
   pour éviter que chaque bibliothèque monopolise tous les cœurs.
4. Ordonnanceur par device : sur un GPU, concurrence bornée et lots mesurés ;
   sur plusieurs GPU, affectation configurable. Le nombre de tâches simultanées
   ne constitue pas une preuve d'accélération. Garder un mode séquentiel témoin.
5. Tracking, propagation terrain, votes OCR et fenêtres balle restent ordonnés
   par séquence. Pas de découpage arbitraire en chunks qui détruirait leur état.
6. Les résultats portent l'identité de leur observation source ; assembler avec
   un tampon de remise en ordre borné. Ne jamais associer un résultat tardif à
   la frame courante. Un segment clôturé ne reçoit pas de nouvel état OCR.
7. Fermer les fenêtres balle aux coupes ; gérer explicitement début/fin de vidéo,
   frames sans contexte et fusion des fenêtres recouvrantes.
8. Utiliser une pression inverse : ralentir le producteur quand une file est
   pleine. Toute inférence volontairement omise est tracée, jamais une frame
   silencieusement perdue. Pas de barrière globale inutile pour calculer une
   distance si seul un résultat balle sans incidence sur elle est en retard.
9. Pour comparer les performances, fixer les observations/crops sélectionnés
   en temps vidéo. Les budgets OCR actuels liés au temps de calcul peuvent
   changer les sorties selon le matériel : proposer un mode reproductible et
   distinguer explicitement un éventuel mode adaptatif.
10. Pas de rendu/encodage vidéo par défaut. Mesurer temps de chargement, calcul,
    attente, I/O, débit total, RAM et VRAM sur le pipeline complet.

Base temporelle : timestamps de présentation source lorsque disponibles ;
`frame_index / fps` uniquement pour une source à cadence constante vérifiée ou
un repli explicitement documenté. Prévoir les FPS variables et les timestamps
invalides. Ne pas assimiler temps vidéo et temps de jeu NBA.

## 5. Modèle de données commun

Contrat versionné, JSON natif, coordonnées et unités explicites. Validation
structurelle et contrôles de références entre objets.

| Objet | Informations principales |
| --- | --- |
| `AnalysisRun` | `schema_version`, `run_id`, source, configuration résolue, modèles/empreintes, capacités, statut, timings |
| `VideoSource` | empreinte, dimensions, durée, base temporelle, frames déclarées et décodées |
| `Segment` | plage vidéo, motif de coupure, inclusion statistique, repère terrain propre au segment |
| `FrameObservation` | `frame_index`, `timestamp_s`, `segment_id`, résultats et statut de chaque brique |
| `PersonObservation` | `observation_id`, rôle et confiance, boîte pixels source, `track_id`, pose, maillot, point au sol, position mètres ou `null` |
| `BallObservation` | centre/boîte pixels source, confiance, statut, contexte temporel ; pas de position au sol supposée pour une balle en l'air |
| `CourtCalibration` | repères, homographie image→mètres, convention des axes, âge, qualité, motif d'indisponibilité |
| `Track` | identifiant `(run_id, segment_id, track_id)`, observations, intervalles continus, rôle et ruptures |
| `IdentityAssignment` | identité roster ou anonyme, intervalle d'application, preuves, statut confirmé/ambigu/inconnu/conflit |
| `DistanceStatistic` | sujet, tracks sources, distance observable, durée suivie/mesurée, couverture, ruptures et exclusions |

Le `track_id` n'est ni un nom, ni une identité physique garantie à travers la
vidéo. Sans roster, conserver des sujets anonymes par track. Avec roster,
agréger des segments séparés seulement si les identités sont confirmées et
compatibles. Un numéro seul ne justifie pas une fusion globale.

Une contradiction d'identité ferme l'intervalle attribué et déclenche une
réévaluation conservatrice ; le dernier nom confirmé ne s'applique pas
automatiquement à toute la trajectoire. Deux tracks simultanées attribuées à la
même personne déclenchent un conflit à résoudre, sans double comptage aveugle.
Conserver les contributions par track pour rendre les agrégations auditables.

Statuts distincts : `ok`, `not_detected`, `not_scheduled`, `disabled`,
`unavailable`, `error`. Une confiance absente reste `null` ; les confiances de
détection, de rôle, de pose, d'OCR et la qualité géométrique restent séparées.
Ne pas les transformer en une probabilité de distance correcte non calibrée.

## 6. Première statistique : distance observable

Pour chaque intervalle continu valide d'un joueur :

`distance_m = somme(norm(position_filtrée[i] - position_filtrée[i-1]))`.

Les positions sont des coordonnées métriques projetées sur le parquet, pas des
déplacements en pixels. Réutiliser la convention NBA portée par `court/geometry.py`.

Règles proposées :

- Bas-centre de boîte, ou chevilles valides si disponibles ; enregistrer la
  méthode. Un changement de méthode susceptible de créer un saut doit être
  contrôlé et peut fermer l'intervalle.
- Refuser calibration absente/expirée, pied hors image, projection aberrante,
  extrapolation excessive et identité/rôle incompatibles.
- Filtrer le bruit de localisation avant intégration. Employer un lissage
  temporel et un contrôle des vitesses plausibles, configurables et évalués :
  ni seuil universel inventé, ni addition brute de tous les tremblements.
- Conserver positions brutes et filtrées, paramètres, rejets et causes. Évaluer
  aussi la sous-estimation introduite par le lissage, notamment dans les virages.
- Rompre à chaque coupe, perte de track, trou de mesure, changement de repère,
  conflit d'identité ou saut non plausible. V1 conservatrice : aucune distance
  ajoutée à travers un trou et aucune extrapolation hors champ.
- Les coordonnées peuvent changer d'orientation entre segments caméra : ne
  jamais les relier directement. On peut sommer des distances de segments
  attribués à la même personne sans relier leurs points de bord.
- La durée mesurée est la somme des intervalles temporels effectivement utilisés
  dans la distance. La durée suivie couvre les intervalles où le sujet est suivi
  dans les plages incluses ; documenter précisément son calcul dans le schéma.
- `coverage_ratio = measured_duration_s / tracked_duration_s`, ou `null` si le
  dénominateur est nul. Ce ratio ne mesure pas la part du match réellement jouée.
- Sans paire de positions valide, distance `null` et statut `unavailable` ; zéro
  est réservé à une mesure valide sans déplacement. Statut `partial` si des
  intervalles suivis n'ont pas pu être mesurés.

Limites à afficher dans les métadonnées : distance estimée sur la vidéo,
couverture partielle, contact au sol approximatif (sauts compris), précision
dépendante des modèles et de la calibration. Ne pas promettre une borne basse
mathématique : le bruit peut aussi surestimer la distance.

## 7. Artefacts de sortie

- `statistics.json` : sortie principale, légère, pour le futur outil de visualisation.
- `observations.jsonl` : toutes les observations unifiées, produites progressivement.
- `tracks.jsonl` : trajectoires, liens d'identité et contributions par intervalle,
  pour inspecter ou recalculer les statistiques sans relancer les réseaux.
- `run.json` : source, paramètres, versions, capacités, qualité, timings et liens
  vers les artefacts ; `progress.json` pour l'avancement.

Prévoir la lecture en flux des trajectoires : aucun tableau de toutes les frames
dans le JSON principal. Publication atomique des sorties terminées ; fichiers
partiels et statut d'échec/interruption conservés. Pas de reprise en milieu de
tracking en V1 ; le recalcul des statistiques depuis des observations terminées
est distinct d'une reprise d'inférence.

Extrait illustratif, valeurs fictives, structure à formaliser au lot 1 :

```json
{
  "schema_version": 1,
  "run_id": "analysis-example",
  "status": "completed",
  "metric": "observed_distance",
  "distance_unit": "m",
  "time_basis": "video",
  "scope": "included_video_intervals",
  "players": [
    {
      "subject_id": "anonymous-s0-t12",
      "player_id": null,
      "player_name": null,
      "team_id": null,
      "jersey_number": "23",
      "identity_status": "anonymous",
      "track_refs": [{"segment_id": 0, "track_id": 12}],
      "observed_distance_m": 42.7,
      "tracked_duration_s": 20.0,
      "measured_duration_s": 16.0,
      "coverage_ratio": 0.8,
      "measurement_status": "partial",
      "warnings": ["calibration_gaps", "identity_not_resolved"]
    }
  ]
}
```

Les arbitres et personnes non classées restent dans les observations, sans
statistique de distance joueur. Aucun calcul de vitesse, possession, tir ou
autre statistique métier supplémentaire dans la sortie V1.

## 8. Lots d'implémentation et critères de passage

| Lot | Livraison | Critère de passage |
| --- | --- | --- |
| 1 — Contrats et référence | Schémas, configuration, CLI, lecture vidéo, exemples sans/avec roster, graphe séquentiel avec adaptateurs simulés | Entrée vidéo seule acceptée ; unités, identifiants, timestamps et sorties validés ; erreurs explicites |
| 2 — Adaptateurs réels | Joueurs/rôles/arbitres, balle vidéo, terrain, pose, OCR sans roster ; chargement des modèles isolé si nécessaire | Chaque brique fonctionne sur un extrait réel sans annotations/dataset imposé ; capacités exactes documentées |
| 3 — Fusion temporelle | Segments communs, tracking, projection, observations, identité anonyme/roster, export en flux | Associations cohérentes ; aucune fusion par simple numéro ambigu ; bornes mémoire et coupes vérifiées |
| 4 — Distance | Trajectoires filtrées, intervalles valides, agrégation, couverture, JSON final, recalcul depuis observations | Distance analytique correcte ; absence de mesure distincte de zéro ; trous/coupes exclus ; identité et double comptage contrôlés |
| 5 — Parallélisme mesuré | Workers persistants, files bornées, lots, budgets CPU/GPU, collecte ordonnée | Sorties équivalentes au témoin à observations fixées ; gain mesuré sur même matériel ou choix du mode séquentiel si plus rapide |
| 6 — Validation et documentation | Tests d'intégration, extraits réels, benchmark, guide d'utilisation, exemple JSON | Vidéo seule et vidéo+roster validées ; rapport de couverture/erreurs/performance ; non-régression des parcours existants |

Le graphe et les interfaces asynchrones sont préparés dès les premiers lots ;
le témoin séquentiel permet d'isoler ensuite les erreurs de concurrence.

**Point de passage du lot 2 : les arbitres.** Un contrat et un champ `referee`
ne constituent pas une détection opérationnelle. Identifier un checkpoint
compatible déjà disponible, ou intégrer une baseline provisoire et l'évaluer.
Si aucune n'est disponible, documenter le besoin de poids/annotations ou d'un
petit entraînement dédié. Ce manque ne bloque pas les autres lots, mais le
pipeline à six capacités ne sera pas déclaré terminé tant qu'il subsiste.
Le choix d'une baseline n'engage pas le choix des modèles définitifs.

Validation fonctionnelle ciblée :

- Déplacement synthétique connu, joueur immobile avec bruit, trajectoire courbe,
  déplacement caméra avec joueur immobile, calibration absente et changement
  de convention entre segments.
- Coupes, trou de détection, échange d'IDs, mêmes numéros entre équipes, OCR tardif,
  contradiction d'identité, doublon de tracks et présence d'arbitres/public.
- Files saturées, résultats hors ordre, erreur de worker, fin de vidéo, interruption,
  frames variables et métadonnées de conteneur inexactes.
- Fenêtres balle aux frontières des segments ; absence de fuite du futur d'un
  segment vers le précédent.
- Petit jeu d'extraits annotés manuellement pour inspecter rôles, identité,
  positions et distances. Fixer les seuils de qualité à partir de ce jeu ;
  tant qu'il manque, distinguer validation technique et précision réelle.

Benchmark : mêmes vidéos, poids, résolutions, cadences et crops, rendu désactivé ;
comparer séquentiel/parallèle, chargement séparé puis temps total, mémoire maximale
et couverture identique. Répéter les mesures pour estimer leur variabilité.
Aucun objectif temps réel ni facteur d'accélération garanti avant mesure.

## 9. Décisions proposées à valider

1. V1 locale hors ligne, entrée vidéo et roster explicitement facultatif.
2. Six capacités intégrées, avec travail supplémentaire identifié pour les arbitres.
3. Distance observable en mètres avec couverture, par track anonyme et, quand
   possible, par identité confirmée ; aucune promesse de réidentification globale.
4. Extraits continus comme référence initiale ; gestion des coupes et exclusions
   connues, sans détection automatique des replays en V1.
5. `statistics.json` comme sortie principale, observations/trajectoires JSONL
   pour audit et future visualisation.
6. Exécution concurrente limitée par les ressources et validée par benchmark,
   modèles interchangeables, DVC et visualisation hors périmètre.

L'implémentation est dans `pipeline/`. Les modèles restent interchangeables ;
les réglages et limites mesurés sont documentés dans son guide et sa validation.
