# Salle vidéo / Film room

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

Les trois panneaux gardent une place stable : **le match à gauche**, **la vue
tactique en haut à droite**, **les statistiques en bas à droite**.
La barre de marque a été supprimée ; FR/EN se trouve à côté du sélecteur d’extrait.
Le titre affiche les équipes de l’effectif. L’identité visuelle utilise un fond
charbon, un accent orange et une typographie de titre condensée.

**Calques** ouvre les réglages détaillés ; le nombre à côté indique les catégories
actuellement affichées. Les cases indiquent l’état de chaque option. Échap ou un
clic à l’extérieur ferme le panneau. **Rétablir l’affichage par défaut** réactive
les Bbox et étiquettes joueurs/arbitres, le ballon et sa traînée, et masque
les poses et les repères détectés. Les lignes projetées du terrain restent
activées. **Agrandir** replie les deux panneaux de droite.

Le bouton **FR / EN** traduit les commandes, descriptions, états, statistiques,
formats numériques et dates, sans recharger la vidéo ni changer la lecture,
le joueur sélectionné ou les réglages. Le choix est conservé localement quand
le navigateur autorise le stockage. Les noms d’équipes, de joueurs et d’analyses
restent les données d’origine. Aucun service de traduction distant n’est utilisé.

La disposition desktop tient dans la fenêtre à partir de 1024 × 640. Les listes
longues et les fiches défilent dans leur panneau. Sous 900 px de largeur, les mêmes
panneaux passent sous la vidéo et la page défile ; les contrôles restent identiques.
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

**Joueurs identifiés** restreint la vidéo et le terrain aux joueurs reconnus par
le run ; les options arbitres sont alors désactivées et une explication apparaît.

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

Les deux onglets du panneau **Statistiques** partagent le même sélecteur d’équipe.
Ils occupent la même zone sans ajouter de panneau sur la vidéo ou le terrain.
Flèches gauche/droite, Début et Fin permettent de naviguer entre les onglets au
clavier. Le choix de langue conserve l’onglet et l’équipe sélectionnés.

**Équipe** affiche le nom et le nombre de joueurs de l’effectif sélectionné,
puis un accès aux heatmaps collectives en attaque ou en défense. La zone `#teamStatsContent` reste défilante
pour accueillir d’autres groupes de statistiques. Les cartes d’équipe cumulent
les contributions observées ; elles ne supposent pas que les cinq joueurs sont visibles.

**Joueurs** conserve la fiche du joueur quand on consulte l’onglet Équipe puis
revient. Changer d’équipe efface la sélection d’un joueur de l’autre équipe.
Sélectionner un joueur depuis la vidéo ou le terrain ouvre automatiquement sa
fiche dans l’onglet Joueurs. Une nouvelle analyse revient à la liste des joueurs.

Choisir une équipe, puis un joueur, ouvre son **bilan de l’extrait complet**.
Le groupe **Déplacements** présente la distance observée, la durée mesurée et la
couverture du suivi issues de `statistics.json`, indépendamment de la lecture.
La flèche **Effectif** revient à la liste. La fiche défile pour pouvoir accueillir
d’autres groupes de statistiques sans réduire le lecteur. Dans les fenêtres de
faible hauteur, les trois mesures se présentent sur une même ligne.

La couverture rapporte la durée mesurée aux intervalles suivis ; elle ne mesure
pas la présence réelle hors champ. Les valeurs absentes restent `—`, et une
distance nulle reste zéro. Aucun chiffre n’est inventé pour compléter une fiche.

Sur les anciens runs sans numéros dans `statistics.json`, les numéros confirmés
sont relus dans `observations.jsonl` par track et segment. Sans effectif de run,
un `jersey.json` voisin de la vidéo peut fournir un effectif indicatif, sans lui
attribuer de statistiques. Un numéro unique peut suggérer une équipe et sa
couleur : la boîte reste pointillée car cette association n’a pas été validée
par le pipeline. Un numéro partagé par les deux équipes reste ambigu.

## Heatmaps : cinq angles à comparer

**Heatmaps** dans l’en-tête du terrain ouvre un espace agrandi, sans modifier les
panneaux du lecteur. Les fiches **Joueurs** et **Équipe** proposent aussi un accès
direct au sujet sélectionné. La vidéo se met en pause. Échap ou le bouton de
fermeture rend le focus au bouton d’origine ; la lecture ne redémarre pas seule.

Choisir un **joueur ou une équipe**, puis **Attaque** ou **Défense**. Les cartes
ne retiennent que le jeu installé dans un camp. Un changement de possession
confirmé termine la phase ; la récupération sous son propre panier et la
remontée du terrain restent neutres jusqu’à l’arrivée dans le camp attaqué.
Ces intervalles sont exclus des cartes, des passages et des libellés du viewer.

La barre Attaque / Défense indique des **secondes de vidéo**, distinctes des
secondes-joueurs collectives. Le quart-temps, les passages et les bornes temporelles
permettent de cibler un moment. Les passages sont continus : ils ne regroupent pas
deux attaques séparées par une transition. **Revoir les passages** ferme la
fenêtre et positionne la vidéo, en pause, au début du passage choisi.

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
autres vues. Passer à Comparaison depuis un passage isolé revient à tous les
passages du quart-temps choisi ; les bornes saisies manuellement restent conservées
si aucun passage n’était sélectionné.

La comparaison applique la situation à **chaque équipe**. Deux équipes en Attaque
sont comparées sur leurs propres passages offensifs, à des moments potentiellement
différents dans la période choisie. Une période limitée à l’attaque d’une seule
équipe peut donc laisser l’autre carte vide. L’évolution partage la période source
en deux moitiés égales et conserve seulement les instants de la situation choisie.
Le choix de langue conserve les filtres, le mode et les résultats.

| Angle | Lecture et intérêt |
| --- | --- |
| **Présence** | Part du temps localisé dans chaque maille : repérer les espaces occupés par un joueur ou la répartition collective. |
| **Zones** | Le temps de présence réparti sur une grille 3 × 3 : une lecture chiffrée plus simple que les taches de chaleur. Les tiers vont de la ligne médiane au panier du demi-terrain affiché. |
| **Comparaison** | Les deux équipes du match, puis leur différence. Chaque profil est normalisé à 100 % de son temps localisé dans le demi-terrain ; les deux premières cartes partagent une échelle. Orange : A davantage représenté ; bleu : B. L’écart est en points de pourcentage, pas une mesure de domination. |
| **Évolution** | La période choisie divisée en deux durées égales. Chaque moitié est normalisée séparément et les deux cartes partagent une échelle : voir un changement d’occupation dans la situation choisie. |

Le rendu **Zones continues**, actif par défaut, regroupe les positions proches
avant d’appliquer des paliers de couleur aux contours nets. Les zones les plus
claires indiquent les concentrations les plus fortes, sur un fond ardoise.
Désactiver cette option affiche les mailles brutes d’environ un mètre ; la vue
Zones conserve sa grille 3 × 3.

Les cartes continues expriment une concentration du temps localisé dans le
demi-terrain en **% par m²**. L’écart de comparaison est en
points de pourcentage par m². Les cartes comparées partagent la même échelle
calculée sur cette concentration, avec un traitement spatial identique.
Le survol et les tableaux conservent les mesures brutes par maille ou par zone ;
les totaux et couvertures ne changent pas avec le rendu. Les cartes PNG exportées
indiquent le sujet, le mode,
la situation, le quart-temps éventuel, la période, la légende et les limites de lecture. Pour une carte simple,
elles conservent aussi le total et la couverture ; une différence indique A et B.
Le tableau sous les cartes rend les neuf zones lisibles au clavier, y compris
les écarts de comparaison.

### Ce que les chiffres mesurent

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
terrain et les contributions de distance ; le viewer propose les quatre vues
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
| Plan précédent / suivant | P / N |
| Fermer les réglages de calques | Échap |

Les raccourcis de lecture ne capturent pas les touches des boutons, sélecteurs,
cases ou du bouton de dépliage des calques. Vitesse et plein écran sont accessibles
sous la vidéo.

## Vérifier

Sans dépendances JavaScript à installer pour les vérifications unitaires :

```bash
python -m unittest viewer.test_server viewer.test_heatmaps viewer.test_phases viewer.test_scoreboard viewer.test_settled viewer.test_shots -v
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
huit résolutions, les contrôles au clavier, les contrastes et libellés, les onglets Joueurs/Équipe,
les termes Bbox/Segmentation dans les deux langues, les calques,
la langue persistante pendant la lecture, la sélection joueur, les listes longues,
les données absentes et le repli sans callbacks de présentation. Un second scénario
couvre les cinq angles pour joueur et équipe dans les deux situations, les périodes,
les passages, les transitions masquées et les exports,
la traduction dans la fenêtre, les réponses réseau tardives et les comparaisons
sans données. Le basculement entre zones continues et mailles vérifie que les
mesures restent identiques, sans requête supplémentaire. Il vérifie aussi
l’accessibilité des cartes et des tableaux. Les contrôles automatiques d’accessibilité ne remplacent pas une revue manuelle.

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
chronomètre du bandeau. Les points turquoise permettent de rejoindre les mises à
jour du score. Ils n’indiquent pas l’instant exact d’un panier : le bandeau peut
s’animer après l’action et le montage peut supprimer du jeu. Le temps de la
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
**Passage de jeu** précise ensuite l’attaque ou la défense. Les intervalles sont
interrompus aux limites de QT ; la plage d’un QT ne déborde pas sur le suivant.
Les quatre vues et la
normalisation du panier à droite restent disponibles.

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

Le volet **Tirs**, sous les commandes vidéo, liste les tentatives estimées. Les
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
