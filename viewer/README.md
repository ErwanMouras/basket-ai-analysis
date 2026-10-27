# Match Analyse

Studio local en lecture seule pour explorer les analyses terminées du pipeline :
vidéo source, détections, positions projetées, statistiques finales des joueurs et
heatmaps individuelles ou collectives.

## Démarrer

Depuis la racine du dépôt :

```bash
python -m viewer.server
```

Ouvrir <http://127.0.0.1:8765>. Le sélecteur recense les analyses terminées sous
`runs/analysis`. L’extrait le plus long s’ouvre par défaut ; `?run=...` permet
un accès direct à une analyse. Pour le sample public, synthétique et scripté :

```bash
make demo-native
```

Il est servi sur <http://127.0.0.1:8766>, sans GPU ni téléchargement de modèles.

Pour utiliser d’autres dossiers :

```bash
python -m viewer.server --runs-root /chemin/vers/runs --video-root /chemin/vers/videos --port 8765
```

`--video-root` délimite les vidéos accessibles au serveur. Le chemin vient de
`source.path` dans `run.json`. Chaque analyse terminée doit contenir `run.json`,
`observations.jsonl` et `statistics.json` ; `tracks.jsonl` fournit les distances
progressives.

## Se repérer

Deux modes se partagent l’espace sous le titre du match : **Visualisation**
réunit la vidéo, les calques, le terrain tactique et la timeline des événements ;
**Statistiques** donne toute la largeur aux bilans **Joueurs / Équipe**.
Le sélecteur reste visible et les deux vues glissent latéralement en 420 ms.
La préférence système de réduction des animations supprime cette transition.
Les flèches gauche/droite, Début et Fin naviguent aussi entre les modes au clavier.
Le mode inactif est retiré de la navigation et de l’arbre d’accessibilité.

Ouvrir Statistiques met la vidéo en pause. Revenir à Visualisation conserve sa
position, sa vitesse et ses calques, ainsi que l’équipe et le joueur sélectionnés ;
la lecture ne redémarre pas seule. Le changement de mode ne recharge pas la vidéo.
FR/EN se trouve à côté du sélecteur d’extrait.
L’onglet du navigateur s’appelle **Match Analyse**. L’en-tête affiche directement
les équipes de l’effectif, sans surtitre ni ligne de métadonnées. L’identité visuelle utilise un fond
charbon, un accent orange et une typographie de titre condensée.

Les réglages **Calques** sont directement visibles dans la barre **Le match**,
entre **Le match** et **Agrandir** : joueurs, arbitres, ballon et terrain.
Les options pose, noms, traînée et repères portent des libellés textuels.
La barre utilise la largeur disponible et les réglages passent à la ligne si
nécessaire, sans défilement horizontal. Sur les petits panneaux, ils occupent
toute la largeur sous le titre. Le filtre Personnes visibles a été retiré.
Aucun menu n’est à ouvrir. Le compteur de catégories a été retiré de cette barre.
Le bouton **Réinitialiser**, centré en hauteur, réactive les Bbox et étiquettes joueurs/arbitres,
le ballon et sa traînée, ainsi que les lignes projetées du terrain ; poses et
repères détectés sont masqués.

**Agrandir** utilise toute la fenêtre pour la vidéo, en largeur et en hauteur.
Les noms d’équipe et la barre **Le match**, avec les calques modifiables,
restent au-dessus de la vidéo. Le nom du fichier a été retiré de cette barre.
Le sélecteur d’extrait, les onglets de
mode et les panneaux latéraux sont masqués. La timeline, sa légende et les commandes
restent visibles en bas sans défilement, y compris sur téléphone.
**Réduire**, dans la barre du lecteur, ou **Échap** rétablit la vue
normale. La lecture, la position, la vitesse et les calques restent conservés.
Le bouton de plein écran du bas a été retiré. Ouvrir une fiche joueur depuis la
vidéo quitte automatiquement l’affichage agrandi.
La ligne d’informations sous le lecteur (état du terrain, détections et catégories)
a été retirée. Les espacements de la timeline, de sa légende et des commandes sont
légèrement réduits pour laisser plus de hauteur à la vidéo.

Le bouton **FR / EN** traduit les commandes, descriptions, états, statistiques,
formats numériques, sans recharger la vidéo ni changer la lecture,
le joueur sélectionné ou les réglages. Le choix est conservé localement quand
le navigateur autorise le stockage. Les noms d’équipes, de joueurs et d’analyses
restent les données d’origine. Aucun service de traduction distant n’est utilisé.

La disposition desktop tient dans la fenêtre à partir de 1024 × 640. Les listes
longues et les fiches défilent dans leur panneau. Sous 900 px de largeur, les mêmes
vues se réorganisent et la page défile : terrain sous la vidéo, effectif au-dessus
de la fiche sur téléphone ; les contrôles restent identiques.
Il n’y a pas de version mobile distincte. Le navigateur peut aussi défiler quand
la hauteur disponible est inférieure à 640 px.

## Calques et synchronisation

**Joueurs** et **Arbitres** ont chacun un choix **Bbox**, **Segmentation**,
**Bbox + Segmentation** ou **Masqués**, plus leurs propres cases **Pose** et
**Noms / numéros**. Les termes Bbox et Segmentation restent identiques en français
et en anglais. Les étiquettes peuvent accompagner les masques seuls. Un ancien
run sans masques n’affiche aucune bbox de repli en mode Segmentation.

**Ballon** commande son affichage et sa traînée sur les cinq images précédentes,
limitée au plan courant. **Terrain** propose deux options indépendantes :
**Lignes projetées**, actives par défaut, dessine les lignes NBA à partir de la
calibration de chaque image, même entre deux passages du détecteur. Un contour
sombre renforce le contraste et le tracé reste cyan pendant le suivi pour éviter
le clignotement. La zone légèrement teintée montre le support des repères ; les
lignes au-delà de cette zone sont extrapolées. Une calibration indisponible ou
invalide efface immédiatement ces lignes.

**Repères détectés**, masqués par défaut, montre uniquement les prédictions de
l'image courante, numérotées de 1 à 18 : cyan pour les points retenus par un
recalage accepté, rose pour les autres points de confiance ≥ 0,5. Ils ne sont
pas prolongés artificiellement entre deux détections. L'état sous la vidéo
continue de distinguer terrain détecté, suivi et indisponible.

La vue tactique montre seulement les personnes ayant une `position_m` valide.
Sa légende reprend les couleurs des équipes. Le ballon reste dans la vidéo :
les artefacts ne fournissent pas sa projection métrique au sol.

Les calques suivent le temps de l’image présentée, y compris en pause et après
un déplacement dans la vidéo. Les observations sont préchargées par paquets ; les
calques sont effacés si l’observation de l’image présentée manque. Le cache de
décodage RLE est borné.

Quand `requestVideoFrameCallback` est disponible, son `mediaTime` commande les
calques. L’événement `seeking` peut arriver après la présentation de la nouvelle
image, notamment après redimensionnement : cette image déjà présentée est
conservée si elle correspond à la destination. Sans cette API, le studio attend
`seeked` et utilise le dernier timestamp inférieur ou égal au temps vidéo.
Les masques stockés restent découpés aux bbox du pipeline ; voir l’[audit de
segmentation](../reports/segmentation-display-audit-2026-10-25/README.md).

## Statistiques : Joueurs / Équipe

Les deux onglets du mode **Statistiques** partagent le même sélecteur d’équipe.
Ils occupent l’espace complet dédié aux statistiques.
Flèches gauche/droite, Début et Fin permettent de naviguer entre les onglets au
clavier. Le choix de langue conserve l’onglet et l’équipe sélectionnés.

Le sélecteur affiche directement les grands noms d’équipe et leurs effectifs,
sans second en-tête. L’équipe sélectionnée occupe **75 %** de la largeur, l’autre
**25 %**. Ces proportions s’inversent avec une transition à chaque sélection.

**Équipe** affiche directement les heatmaps collectives en attaque ou en défense,
avec les vues Présence et Zones. Le petit bouton **Comparer** affiche les deux
équipes et leur différence ; un second clic rend la vue précédente.
Changer d’équipe conserve la situation de jeu et le quart-temps.
Les boutons **Attaque / Défense** sont placés sous le terrain du cadre Présence et ne filtrent
que les heatmaps de présence ou de zones, y compris pendant la comparaison.
La carte des tirs conserve le bilan de tout l’extrait.
La zone `#teamStatsContent` reste défilante ; présence et tirs s’affichent côte à côte,
puis se superposent sur téléphone. Les terrains occupent la largeur disponible
jusqu’à 560 px ; sur desktop, leur taille s’adapte à la hauteur du panneau pour
afficher les deux terrains alignés sans défilement. Les en-têtes et espacements
sont réduits dans cette vue. Les cartes utilisent un rendu haute définition,
sans les libellés demi-terrain et panier. Les zones continues sont actives par défaut.
Les cartes cumulent les contributions observées et ne supposent pas que les cinq
joueurs sont visibles.

**Joueurs** conserve la fiche du joueur quand on consulte l’onglet Équipe puis
revient. À l’entrée dans les statistiques des joueurs, le premier joueur de
l’équipe est présélectionné lorsqu’aucune fiche n’est encore choisie. Une sélection
manuelle reste conservée lors des allers-retours. Changer d’équipe dans l’onglet
Joueurs ouvre la première fiche du nouvel effectif.
Sélectionner un joueur depuis la vidéo ou le terrain ouvre automatiquement sa
fiche dans l’onglet Joueurs du mode Statistiques, avec la vidéo en pause.
Une nouvelle analyse revient à l’onglet Joueurs dans le mode courant ; si les
statistiques sont déjà ouvertes, la première fiche disponible est sélectionnée.

Choisir une équipe, puis un joueur, ouvre son **bilan de l’extrait complet**.
Le groupe **Déplacements** présente la distance observée, la durée mesurée et la
couverture du suivi issues de `statistics.json`, indépendamment de la lecture.
L’effectif reste accessible à gauche de la fiche sur desktop ; la sélection y est
signalée. La flèche **Effectif** ferme la fiche. Les listes et les fiches défilent
indépendamment. Les trois mesures se présentent sur une même ligne.

La couverture rapporte la durée mesurée aux intervalles suivis ; elle ne mesure
pas la présence réelle hors champ. Les valeurs absentes restent `—`, et une
distance nulle reste zéro. Aucun chiffre n’est inventé pour compléter une fiche.

Sur les anciens runs sans numéros dans `statistics.json`, les numéros confirmés
sont relus dans `observations.jsonl` par track et segment. Sans effectif de run,
un `jersey.json` voisin de la vidéo peut fournir un effectif indicatif, sans lui
attribuer de statistiques. Un numéro unique peut suggérer une équipe et sa
couleur : la boîte reste pointillée car cette association n’a pas été validée
par le pipeline. Un numéro partagé par les deux équipes reste ambigu.

## Heatmaps

Les fiches **Joueurs** proposent un accès aux heatmaps du joueur sélectionné dans
une fenêtre agrandie. Le bouton Heatmaps a été retiré de la vue tactique. Les heatmaps d’équipe sont intégrées à l’onglet **Équipe**
avec des filtres indépendants de la fenêtre agrandie. La vidéo se met en pause. Échap ou le bouton de
fermeture rend le focus au bouton d’origine ; la lecture ne redémarre pas seule.

Choisir un **joueur ou une équipe**, puis **Attaque** ou **Défense**. Les cartes
ne retiennent que le jeu installé dans un camp. Un changement de possession
confirmé termine la phase ; la récupération sous son propre panier et la
remontée du terrain restent neutres jusqu’à l’arrivée dans le camp attaqué.
Ces intervalles sont exclus des cartes, des passages et des libellés du viewer.

Les boutons **Attaque / Défense** changent la situation affichée. Leurs durées,
quand elles sont disponibles, indiquent des **secondes de vidéo**, distinctes des
secondes-joueurs collectives. Le sélecteur **Quart-temps** couvre la période choisie ;
**Tout l’extrait** reprend l’extrait entier. Les filtres Passage de jeu, Début et
Fin ont été retirés. Les cartes gardent leur légende, sans bandeau de mesures,
survol chiffré, tableau par zone, relecture de passages ni bloc explicatif sous
les cartes. Les états sans données restent explicites.

### Comment les phases installées sont estimées

Le sens des paniers est appris à partir des contacts balle–mains et de la position
d’au moins deux défenseurs, puis conservé sur le quart-temps. Un angle inversé
confirmé garde son repère propre ; sans QT lisible, le repère local du plan peut
servir. Le ballon n’est pas projeté au sol.

Un état simple distingue une attaque installée de chaque équipe et une transition
neutre. Les contacts répétés confirment le contrôle dans une fenêtre passée de
0,4 s (au moins 0,16 s sur deux images, 80 % d’accord). Un nouveau contrôleur
termine l’attaque précédente ; il ne démarre pas son attaque sous son propre
panier. Il faut atteindre son demi-terrain attaqué, par un porteur localisé au-delà
de la bande centrale ou un regroupement spatial stable.

Une attaque établie survit à une disparition du ballon. Les votes de contact et
de terrain repartent à zéro aux coupes et trous du suivi ; une transition déjà
confirmée reste toutefois en attente du même nouveau contrôleur après une coupe
continue du même QT. Une calibration non exploitable ou un contact ambigu masque
le libellé. Une mise à jour observée du score termine la phase courante sans
attribuer automatiquement l’attaque à l’autre équipe, et sans antidater le panier.

Sous la vidéo, **« LAL attaque · HOU défense »** n’apparaît que pendant une phase
installée ; aucun libellé neutre ne le remplace. Les durées internes non retenues
restent disponibles pour les audits, sans être proposées comme vue d’analyse.

Les passages exploitables sont ensuite remis dans le même sens, avec une rotation
de 180° si nécessaire : **panier attaqué à droite en Attaque**, **panier défendu à
droite en Défense**. Les cartes peuvent alors cumuler plusieurs plans sans
superposer des sens opposés. Gauche et droite correspondent aux côtés en regardant
vers ce panier de référence. Sans repère de panier exploitable, aucun intervalle
n’entre dans une carte Attaque ou Défense.

Toutes les cartes affichent uniquement le **demi-terrain du panier de référence**.
Après la rotation, le calcul conserve les positions entre la ligne médiane et ce
panier. Un trajet traversant la ligne est découpé à son instant de franchissement :
seul le temps passé dans ce camp contribue à la grille, aux neuf zones et aux
pourcentages. Les positions dans l’autre moitié ne sont ni repliées ni comptées
comme des trous de calibration. Une note indique leur temps exclu lorsqu’il existe.
Le terrain NBA affiché mesure 14,3256 × 15,24 m ; le sens du panier reste à droite.

La comparaison utilise automatiquement les **deux équipes de l’effectif**, sans
sélecteur de sujet ou d’adversaire. Le joueur sélectionné reste mémorisé pour les
autres vues. La comparaison conserve la situation et le quart-temps choisis.

La comparaison applique la situation à **chaque équipe**. Deux équipes en Attaque
sont comparées sur leurs propres passages offensifs, à des moments potentiellement
différents dans la période choisie. Une période limitée à l’attaque d’une seule
équipe peut donc laisser l’autre carte vide. Le choix de langue conserve les
filtres, le mode et les résultats.

| Angle | Lecture et intérêt |
| --- | --- |
| **Présence** | Part du temps localisé dans chaque maille : repérer les espaces occupés par un joueur ou la répartition collective. |
| **Zones** | Le temps de présence réparti sur une grille 3 × 3 : une lecture chiffrée plus simple que les taches de chaleur. Les tiers vont de la ligne médiane au panier du demi-terrain affiché. |
| **Comparaison** | Les deux équipes du match, puis leur différence. Chaque profil est normalisé à 100 % de son temps localisé dans le demi-terrain ; les deux premières cartes partagent une échelle. Orange : A davantage représenté ; bleu : B. L’écart est en points de pourcentage, pas une mesure de domination. |

Le rendu **Zones continues**, actif par défaut, regroupe les positions proches
avant d’appliquer des paliers de couleur aux contours nets. Les zones les plus
claires indiquent les concentrations les plus fortes, sur un fond ardoise.
Désactiver cette option affiche les mailles brutes d’environ un mètre ; la vue
Zones conserve sa grille 3 × 3.

Les cartes continues expriment une concentration du temps localisé dans le
demi-terrain en **% par m²**. L’écart de comparaison est en
points de pourcentage par m². Les cartes comparées partagent la même échelle
calculée sur cette concentration, avec un traitement spatial identique.
Les exports PNG ont été retirés des heatmaps et des cartes de tirs.

### Mesures internes de l’API

- **Temps dans ce camp** : somme des durées localisées dans le demi-terrain
  affiché, après orientation et découpage à la ligne médiane. Pour une équipe,
  deux joueurs suivis pendant une seconde représentent **deux secondes-joueurs**.
- **Suivi localisable** : temps localisé et orienté / temps de suivi attribué au
  sujet dans cette situation. Les positions valides dans l’autre moitié restent
  localisées pour ce taux de qualité, mais ne contribuent pas à la carte. Les
  absences, identités inconnues et joueurs hors champ n’entrent pas au dénominateur.
- **Vidéo dans ce camp** : union des intervalles comportant au moins une position
  localisable dans le demi-terrain affiché. Pour une équipe, cela n’implique pas
  un suivi complet des cinq joueurs.

L’API expose `half_court_s`, `outside_half_s` et `half_court_video_s` pour séparer
le recadrage spatial de la qualité de localisation. Les requêtes de phases
retournent `court.extent: "half"`, la demi-longueur affichée et `court.full_length`.
Le chemin d’audit historique sans filtre de phase conserve ses coordonnées plein
terrain et les contributions de distance ; le viewer propose les trois vues
ci-dessus. La comparaison API accepte uniquement les deux équipes du match et
choisit l’adversaire automatiquement.

Le calcul pondère par le temps, jamais par le nombre d’images. Il coupe aux
changements de plan, frames manquantes, exclusions, grands intervalles de temps,
changements d’identité et changements de méthode de point au sol. Les vitesses
incompatibles avec le seuil du run sont écartées. Il n’extrapole pas après la
dernière observation. Les petites portions de trajet sont interpolées à l’intérieur
des seuls intervalles valides et réparties sur la grille en conservant le total.

Le découpage nécessite exactement deux équipes dans l’effectif. La reconnaissance
d’équipe et la calibration peuvent se tromper ; les pauses, remises en jeu,
transitions et ralentis peuvent aussi affecter l’interprétation. Une phase classée
ne prouve pas une possession officielle ou une attaque placée complète.

Il faut des dimensions métriques explicites dans `run.json` et un véritable
`roster.json` du run. Un effectif indicatif n’attribue aucune heatmap nominative.
Sans données pour l’un des deux sujets comparés, aucune différence n’est produite.
L’index compact est conservé pour trois runs au maximum et invalidé quand les
observations, le manifeste, l’effectif ou les contributions de distance changent.

Les cartes décrivent **les positions observées**. Une zone sombre ne prouve pas
qu’elle était libre ; un suivi partiel peut orienter la répartition. La précision
métrique n’est pas validée par ces cartes et les ralentis ne sont pas exclus
automatiquement. Le ballon n’a pas de projection métrique exploitable ici : il
ne contribue pas aux cartes de présence. Les phases installées restent des
estimations conservatrices, et non des possessions officielles.

## Raccourcis

| Action | Raccourci |
| --- | --- |
| Lire / mettre en pause | Espace |
| Reculer / avancer d’une seconde source | ← / → |
| Image précédente / suivante | Maj + ← / → |
| Réduire la vidéo agrandie | Échap |

Les raccourcis de lecture ne capturent pas les touches des boutons, sélecteurs,
cases de calques. La vitesse est accessible sous la vidéo.

## Vérifier

Sans dépendances JavaScript à installer pour les vérifications unitaires :

```bash
python -m unittest viewer.test_server viewer.test_heatmaps viewer.test_phases viewer.test_scoreboard viewer.test_settled viewer.test_shots viewer.test_shot_analytics -v
node --check viewer/app.js
node --check viewer/i18n.js
node --check viewer/heatmaps.js
node --check viewer/heatmap-renderer.js
node --test viewer/test_app.js viewer/test_heatmap_renderer.cjs
```

Pour les contrôles navigateur, démarrer le sample avec `make demo-native`, puis :

```bash
npm --prefix viewer ci
node viewer/node_modules/playwright/cli.js install chromium
npm --prefix viewer run test:browser
```

Le test utilise Chromium et axe-core uniquement pour le développement. Il couvre
neuf résolutions, les contrôles au clavier, les contrastes et libellés, les onglets Joueurs/Équipe,
les deux modes, le glissement latéral et sa réduction, la pause à l’ouverture des
statistiques et la conservation du contexte entre les modes,
les termes Bbox/Segmentation dans les deux langues, les calques,
la langue persistante pendant la lecture, la sélection joueur, les listes longues,
les données absentes et le repli sans callbacks de présentation. Un second scénario
couvre les trois angles pour joueur et équipe dans les deux situations, les périodes
et les transitions masquées,
la traduction dans la fenêtre, les réponses réseau tardives et les comparaisons
sans données. Le basculement entre zones continues et mailles vérifie que les
mesures restent identiques, sans requête supplémentaire. Il vérifie aussi
l’accessibilité des cartes et des contrôles de situation. Le scénario des heatmaps intégrées
vérifie aussi le sélecteur 75/25, la comparaison, les filtres conservés et les
réponses tardives d’une autre équipe. Les contrôles automatiques d’accessibilité ne remplacent pas une revue manuelle.

Options : `VIEWER_URL` pour une autre instance servant le sample,
`VIEWER_SCREENSHOTS` pour enregistrer les captures et
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` pour un Chromium déjà installé.
GitHub Actions rejoue ces contrôles dans le job `viewer-browser` et conserve les
captures du sample public comme artefacts.

La police **Barlow Condensed Bold** est incluse localement, sous
[SIL Open Font License](BarlowCondensed-OFL.txt), depuis le
[dépôt Google Fonts](https://github.com/google/fonts/tree/main/ofl/barlowcondensed).
Le viewer ne charge aucune police, bibliothèque ni image depuis un CDN.

## Reproduire l’expérience de découpage

Le comparateur lit les artefacts terminés et n’exécute aucune inférence :

```bash
python -m viewer.phase_experiment --run runs/analysis/hou-lal-sam-yolo26l-ball \
  --review-checkpoints reports/attack-defense-2026-10-26/review-checkpoints.json \
  --output release/phase-experiment.json
```

Les points de contrôle sont liés au hash de la vidéo HOU–LAL locale. Ils constituent
une petite vérification de développement, sans évaluation indépendante de
précision. Omettre `--review-checkpoints` pour obtenir les durées classées sur un
autre run, par exemple `examples/demo/runs/synthetic`. Le [rapport de
l’expérience](../reports/attack-defense-2026-10-26/README.md) décrit les variantes,
les résultats et leurs limites.

## Bandeau NBA : score, quart-temps et côtés stables

Le lecteur accepte un `scoreboard.json` dérivé, associé au SHA256 de la vidéo et
aux deux équipes de l’effectif. La timeline affiche le score observé, le QT et le
chronomètre du bandeau. Les mises à jour du score ne créent plus de repères sur
la timeline. Le temps de la
vidéo et le chronomètre du match restent deux informations distinctes.

Un champ non lisible reste `—`. Le chronomètre n’est pas extrapolé et une lecture
ne reste pas affichée au-delà de 1,51 intervalle d’échantillonnage. Un run sans
bandeau préparé conserve le lecteur habituel. L’API du viewer reste en lecture
seule, sans OpenCV ni OCR à installer dans son conteneur.

La préparation se fait **hors lecture**, dans l’environnement ML avec OpenCV,
Tesseract et ses données `eng` disponibles dans le PATH :

```bash
.venv-players/bin/python -m viewer.scoreboard_extract \
  --run runs/analysis/hou-lal-sam-yolo26l-ball \
  --left-team houston_rockets:HOU \
  --right-team los_angeles_lakers:LAL
```

Le profil intégré `espn-playoffs-bottom-v1` a été vérifié sur cet habillage ESPN
1920×1080 et se met à l’échelle de la résolution source. La position est fixe
**pour cet habillage** ; les autres diffuseurs NBA ne partagent pas tous ce
bandeau. `--profile autre-profil.json` accepte `name`, `reference_size: [largeur,
hauteur]`, et `fields` : `left_team`, `right_team`, `left_score`, `right_score`,
`period`, `clock`, chacun en rectangle `[x1, y1, x2, y2]` dans l’image de référence.
Les couleurs/prétraitements actuels sont adaptés aux panneaux noir/jaune ESPN :
un autre habillage doit aussi être vérifié avant usage. Les abréviations visibles
sont vérifiées avant d’associer les scores aux équipes ; leur ordre n’est pas
déduit de celui de l’effectif. Une source relative exige `--video /chemin/absolu`.

L’extracteur échantillonne à 2 Hz, exige deux lectures voisines concordantes pour
le QT et pour le score, conserve les lectures brutes et sa provenance, et ignore
les animations masquant les noms d’équipes. Les baisses de score et retours à un
QT antérieur restent indisponibles : corrections et ralentis nécessitent une
vérification humaine. Les ralentis du même QT ne sont pas détectés globalement.
La commande écrit seulement le dérivé `scoreboard.json`, sans relancer les modèles
ni modifier les observations, les annotations ou les poids.

Une fois le QT lu, le premier angle avec au moins **0,75 s** d’indices spatiaux et
**80 %** d’accord fixe son sens de référence. Les autres plans de ce QT le
réutilisent même sans nouvel indice de direction. Un angle inverse exige 0,75 s
et **90 %** d’accord pour être traité séparément ; un angle fortement
contradictoire reste sans orientation. Les courts indices contraires ne
retournent pas tout le QT. Les seuils sont expérimentaux, pas des probabilités.

Les QT restent indépendants : aucune inversion automatique à chaque changement
de période. En NBA, le changement de panier intervient à la mi-temps et les
prolongations conservent le sens de la seconde mi-temps
([règle NBA 4, section I](https://official.nba.com/rule-no-4-definitions/)).
Sans QT confirmé, le repère local par plan reste utilisé. Verrouiller les côtés
ne transforme pas une absence de contrôle de balle en possession connue. Il permet
cependant la classification de contexte décrite ci-dessous.

Dans les heatmaps, **Quart-temps** sélectionne la plage source correspondante ;
les boutons **Attaque / Défense** choisissent la situation. Les intervalles sont
interrompus aux limites de QT ; la plage d’un QT ne déborde pas sur le suivant.
Les trois vues et la normalisation du panier à droite restent disponibles.

L’[expérience sur HOU–LAL](../reports/quarter-score-2026-10-26/README.md) détaille
la couverture des lectures, les points contrôlés et leurs limites. La démo
publique contient un bandeau et une coupe QT1 → QT3 explicitement scénarisés.

### Modèle de phases courant

Le [rapport des phases installées](../reports/settled-play-2026-10-26/README.md)
remplace le précédent complément par zone. Le contrôle du ballon sert à confirmer
les changements ; l’arrivée dans le camp attaqué détermine quand une nouvelle
attaque/défense peut entrer dans les heatmaps. Les transitions restent internes
et ne sont jamais proposées dans le sélecteur ni affichées sous la vidéo.

La [revue des demi-terrains](../reports/half-court-2026-10-26/README.md) décrit
le cadrage final, la comparaison automatique et les vérifications associées.

## Événements de tir

Le terrain tactique occupe une zone réduite en hauteur. Le volet **Tirs**, juste
en dessous, est ouvert par défaut et utilise l’espace restant, sans réduire la vidéo.
Chaque ligne affiche le temps et la frame du tir, le tireur et le statut.
Le bouton **Voir le tir**, comme toute la ligne, place la vidéo en pause avant le relâchement.
La ligne **Interceptions — Arrive bientôt** reste non interactive, sans contenu
ni ouverture possible.

Les commandes vidéo sont centrées : image précédente, −1 s, lecture/pause, +1 s,
image suivante. Le timecode reste à gauche et la vitesse à droite.
Sur téléphone, les commandes centrales passent sur leur propre ligne.

La timeline affiche uniquement les tirs probables et les candidats à vérifier.
Les points sont centrés sur la ligne et mesurent tous 10 px de diamètre.
La légende, placée sous la timeline à gauche, laisse
respirer la zone avant les commandes vidéo. Elle distingue ces deux catégories, avec un
repère creux pour les candidats ; leur filtre met aussi la légende à jour.
Les repères et commandes de changements de plan ont été retirés.

Les
repères violets pleins correspondent aux **tirs probables** ; les repères creux
sont des **candidats à vérifier**. Cliquer sur un repère ou une ligne place la
vidéo en pause 0,6 s avant l’instant estimé du départ. La case **Inclure les
candidats** filtre à la fois la liste et la timeline. La langue conserve ce choix.

Le détecteur lit les observations existantes une fois à l’ouverture du run :
aucune inférence, OCR ou nouvelle dépendance ML ne s’exécute pendant la lecture.
`/api/run` inclut `shots`, également accessible par `/api/shots?name=…`. L’index est
invalidé avec les observations et les métadonnées du run. Un ancien run sans
ballon ou contexte exploitable annonce **données insuffisantes**, sans fabriquer de
zéro officiel.

### Indices croisés

1. Départ du ballon près d’un poignet suffisamment visible, au-dessus des épaules,
   avec plusieurs observations proches dans le même suivi. Les mains concurrentes,
   arbitres et identités contradictoires empêchent une attribution nominative.
2. Ballon observé loin des mains sur au moins 65 % de la courte fenêtre de vol.
   Une réception avant le panier peut écarter une passe ; un ballon tenu ou un
   dribble ne suffit pas.
3. Montée et descente cohérentes, ajustées par une courbe quadratique en temps
   source. La courbure et les écarts sont normalisés par la taille apparente d’un
   joueur, **pas exprimés en m/s²**. La grille du terrain ne projette pas le ballon
   aérien comme s’il touchait le sol.
4. Compensation du mouvement d’image par les homographies du terrain. Ce recalage
   de plan est une approximation pour un ballon en hauteur ; un grand saut de
   trajectoire, une coupe, une exclusion ou un trou supérieur à 0,2 s interrompt
   la fenêtre. Les positions absentes ne sont pas interpolées.
5. Proximité d’un panier élevé estimé à partir d’une caméra plausible et de la
   calibration NBA. Le modèle suppose pixels carrés et point principal au centre
   de l’image ; il renonce à l’ancrage quand ses contraintes ne sont pas plausibles.
   Ce repère approximatif n’est pas une détection visuelle de l’arceau.

Un **tir probable** exige relâchement soutenu, arc propre, proximité du panier et
compensation disponibles. Un arc sans panier fiable, une finition partielle ou un
relâchement masqué reste **à vérifier**. Un deuxième chemin peut proposer un arc
près du panier sans mains associables ; il ne nomme alors aucun tireur et donne
une fenêtre de départ plus large. Les propositions partageant le même vol sont
regroupées. Le tireur n’est nommé que si le roster du run et l’identité résolue le
permettent, sans conflit à proximité du départ.

Le score TV n’est pas utilisé pour créer les tirs. **Marqué/raté, 2/3 points,
lancer franc et statistiques officielles ne sont pas attribués.** Les ralentis,
lobs, contres, dunks et tip-ins très courts peuvent encore tromper ou échapper
au détecteur. Les catégories « probable » et « à vérifier » sont des niveaux de
support heuristique, pas des probabilités calibrées.

### Reproduire une extraction

```bash
python -m viewer.shot_experiment \
  --run runs/analysis/hou-lal-sam-yolo26l-ball \
  --review reports/shots-2026-10-26/review-checkpoints.json \
  --output /tmp/shots.json
```

Sans `--review`, la commande fonctionne sur tout run terminé. `--compact` omet
les points de trajectoire. L’export contient les hashes de la source déclarée,
des observations, du manifeste, du roster et des deux fichiers du détecteur,
ainsi que sa version, ses seuils et les indices de chaque événement. Il ne modifie
aucun artefact d’entrée. Les checkpoints de revue doivent correspondre exactement
aux hashes vidéo et observations.

Voir le [rapport de développement et ses limites](../reports/shots-2026-10-26/README.md).

### Cartes de tirs et compteurs joueurs

**Équipe → Carte des tirs** présente les positions au sol estimées avant les
relâchements, sur un demi-terrain normalisé vers le panier attaqué à droite.
La sélection d’équipe conserve une échelle commune en tirs/m². Les cercles sont
les origines estimées ; le survol lit les timestamps proches ou le nombre par
maille. L’échelle chiffrée de densité et le message sur les tirs sans équipe
ne sont pas affichés.

Les **probables et candidats à vérifier** entrent dans la carte, respectivement
avec un point plein et un point creux. Seuls les probables entrent dans le compteur
**Tirs détectés** de chaque fiche joueur. Le bilan
porte sur l’extrait entier, indépendamment du curseur vidéo et du filtre des
candidats. Une absence de distance ne masque pas le compteur de tirs.

L’origine est la médiane d’au moins deux positions calibrées du tireur dans les
0,25 s avant le départ, sans franchir de coupe. Les changements de méthode et
les dispersions supérieures à 1,25 m sont exclus. Le sens utilise les repères
appris par plan et quart-temps ; il ne dépend pas simplement du côté occupé par
le tireur. Aucune projection du ballon aérien ne remplace une origine manquante.

Un probable ou candidat sans position ou orientation est compté mais absent de la carte.
Les événements sans équipe connue restent hors des cartes et des compteurs
nominatifs. Zéro signifie aucun probable attribué sur l’extrait ; `—` indique
que les données nécessaires ne sont pas disponibles. Aucun taux de réussite
ou statistique officielle n’est ajouté.

Les métadonnées `shots` exposent `player_counts`, `unassigned_probable_count`,
`unassigned_candidate_count` et `shot_maps` (`scope: whole_clip`,
`statuses: [probable, candidate]`). Chaque carte contient `probable_count`,
`candidate_count`, `detected_count`, `located_count`, `unlocated_count`, sa grille
de comptage et ses points avec timestamp et statut. Les événements exposent leur `origin` et leur
`court_direction`. Les statistiques historiques de déplacement restent inchangées.

Voir la [revue des cartes de tirs](../reports/shot-maps-2026-10-26/README.md).
