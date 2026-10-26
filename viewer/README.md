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

Choisir un **joueur ou une équipe**, puis une situation : **Attaque**, **Défense**
ou **Non classé**. Les cartes cumulent par défaut tous les passages classés dans
la situation choisie, sur l’extrait complet. Le sélecteur **Passage de jeu** et les
bornes temporelles permettent ensuite de cibler une période. Un passage peut
contenir de courts trous non classés : ces trous restent exclus du calcul.

La barre Attaque / Défense / Non classé indique des **secondes de vidéo**, distinctes
des secondes-joueurs des cartes collectives. Ses boutons changent la situation.
**Revoir les passages** ferme la fenêtre et positionne la vidéo, en pause, au début
du passage choisi. La vue Non classé expose les intervalles d’au moins 0,2 s à
revoir, sans leur imposer une heatmap orientée.

### Comment le découpage est estimé

Le détecteur recherche des contacts balle–mains en pixels source, avec une
proximité rapportée à la taille du joueur. Sans poignets utilisables, un contact
corporel plus restrictif peut fournir un indice. Un adversaire, un arbitre ou un
joueur non attribué proche de la balle peut bloquer la décision. Le ballon élevé
n’est jamais projeté au sol pour choisir le porteur.

Les contacts doivent se répéter dans une fenêtre temporelle de 0,8 s, avec au
moins 0,16 s de soutien sur plusieurs images et 80 % d’accord entre les contacts attribués. Les indices
ne traversent ni coupes détectées, ni frames absentes, ni périodes exclues ou
grands écarts temporels. Les longues absences de balle restent non classées.
Ce réglage est expérimental : ce n’est ni un pourcentage de confiance, ni un
comptage officiel des possessions.

Le sens est estimé séparément, à partir des contacts en jeu placé et de la position
d’au moins deux défenseurs. Un porteur près d’une ligne de fond ne fixe pas le sens
à lui seul : il pourrait remettre en jeu sous son propre panier. Des indices
insuffisants ou contradictoires laissent le sens inconnu. Les indices sont
réévalués par plan caméra ; aucune orientation n’est transférée d’un plan à l’autre.

Les passages exploitables sont ensuite remis dans le même sens, avec une rotation
de 180° si nécessaire : **panier attaqué à droite en Attaque**, **panier défendu à
droite en Défense**. Les cartes peuvent alors cumuler plusieurs plans sans
superposer des sens opposés. Gauche et droite correspondent aux côtés en regardant
vers ce panier de référence. Les périodes classées mais sans sens établi restent
écartées des cartes et leur durée est signalée.

La comparaison applique la situation à **chaque sujet**. Deux équipes en Attaque
sont comparées sur leurs propres passages offensifs, à des moments potentiellement
différents dans la période choisie. Une période limitée à l’attaque d’une seule
équipe peut donc laisser l’autre carte vide. L’évolution partage la période source
en deux moitiés égales et conserve seulement les instants de la situation choisie.
Le choix de langue conserve les filtres, le mode et les résultats.

| Angle | Lecture et intérêt |
| --- | --- |
| **Présence** | Part du temps localisé dans chaque maille : repérer les espaces occupés par un joueur ou la répartition collective. |
| **Déplacements** | Part des mètres mesurés dans chaque maille : distinguer une zone traversée d’une zone où l’on reste. Ce n’est pas une carte de vitesse ou d’intensité. |
| **Zones** | Le temps de présence réparti sur une grille 3 × 3 : une lecture chiffrée plus simple que les taches de chaleur. Les zones sont géométriques, sans attribution attaque/défense. |
| **Comparaison** | Deux joueurs ou deux équipes, puis leur différence. Chaque profil est normalisé à 100 % de son temps localisé ; les deux premières cartes partagent une échelle. Orange : A davantage représenté ; bleu : B. L’écart est en points de pourcentage, pas une mesure de domination. |
| **Évolution** | La période choisie divisée en deux durées égales. Chaque moitié est normalisée séparément et les deux cartes partagent une échelle : voir un changement d’occupation dans la situation choisie. |

Le rendu **Zones continues**, actif par défaut, regroupe les positions proches
avant d’appliquer des paliers de couleur aux contours nets. Les zones les plus
claires indiquent les concentrations les plus fortes, sur un fond ardoise.
Désactiver cette option affiche les mailles brutes d’environ un mètre ; la vue
Zones conserve sa grille 3 × 3.

Les cartes continues expriment une concentration en **% par m²** : du temps
localisé, ou des mètres mesurés pour Déplacements. L’écart de comparaison est en
points de pourcentage par m². Les cartes comparées partagent la même échelle
calculée sur cette concentration, avec un traitement spatial identique.
Le survol et les tableaux conservent les mesures brutes par maille ou par zone ;
les totaux et couvertures ne changent pas avec le rendu. Les cartes PNG exportées
indiquent le sujet, le mode,
le plan, la période, la légende et les limites de lecture. Pour une carte simple,
elles conservent aussi le total et la couverture ; une différence indique A et B.
Le tableau sous les cartes rend les neuf zones lisibles au clavier, y compris
les écarts de comparaison.

### Ce que les chiffres mesurent

- **Temps localisé** : somme des intervalles avec positions valides aux deux
  extrémités. Pour une équipe, deux joueurs suivis pendant une seconde représentent
  **deux secondes-joueurs**, pas deux secondes de vidéo.
- **Suivi localisable** : temps localisé / temps de suivi attribué au sujet.
  Les intervalles suivis mais non projetables réduisent ce taux. Les absences,
  identités inconnues et joueurs hors champ n’entrent pas dans le dénominateur.
- **Vidéo documentée** : union des intervalles comportant au moins une position
  localisable pour le sujet. Pour une équipe, cela n’implique pas un suivi complet
  des cinq joueurs.
- **Distance répartie** : contributions valides de `tracks.jsonl`, déjà calculées
  par le pipeline, réparties le long des courts trajets correspondants. Une note
  indique la part du temps localisé disposant d’une mesure de distance. Zéro mètre
  mesuré et absence de mesure restent distingués.

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
ne contribue pas aux heatmaps. Aucune possession, phase offensive ou défensive
n’est déduite.

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
python -m unittest viewer.test_server viewer.test_heatmaps viewer.test_phases -v
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
les passages, le retour à la vidéo depuis Non classé et les exports,
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
