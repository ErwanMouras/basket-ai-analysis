# Salle vidéo / Film room

Studio local en lecture seule pour explorer les analyses terminées du pipeline :
vidéo source, détections, positions projetées et statistiques finales des joueurs.

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
ainsi qu’un espace explicite pour les futures statistiques collectives. Aucun
chiffre collectif n’est actuellement calculé ni déduit des statistiques joueur.
La future intégration pourra remplacer le contenu de `#teamStatsContent`, dans
une zone qui défile quand nécessaire, une fois les données d’équipe définies.

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
python -m unittest viewer.test_server -v
node --check viewer/app.js
node --check viewer/i18n.js
node --test viewer/test_app.js
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
les données absentes et le repli sans callbacks de présentation. Les contrôles
automatiques d’accessibilité ne remplacent pas une revue manuelle.

Options : `VIEWER_URL` pour une autre instance servant le sample,
`VIEWER_SCREENSHOTS` pour enregistrer les captures et
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` pour un Chromium déjà installé.
GitHub Actions rejoue ces contrôles dans le job `viewer-browser` et conserve les
captures du sample public comme artefacts.

La police **Barlow Condensed Bold** est incluse localement, sous
[SIL Open Font License](BarlowCondensed-OFL.txt), depuis le
[dépôt Google Fonts](https://github.com/google/fonts/tree/main/ofl/barlowcondensed).
Le viewer ne charge aucune police, bibliothèque ni image depuis un CDN.
