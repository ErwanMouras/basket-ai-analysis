# Studio d'analyse vidéo

Visualiseur local en lecture seule pour les runs terminés du pipeline. Il affiche la vidéo source, les calques de détection (joueurs, arbitres, ballon, pose, identifiants, repères terrain), une vue du terrain projeté et les statistiques finales des joueurs.

## Démarrer

Depuis la racine du dépôt :

```bash
python -m viewer.server
```

Ouvrir <http://127.0.0.1:8765>. Le sélecteur recense les runs terminés présents sous `runs/analysis` avec leur match et leur durée. Le run le plus long s'ouvre par défaut, sauf si `?run=...` précise un run.

Pour utiliser d'autres dossiers :

```bash
python -m viewer.server --runs-root /chemin/vers/runs --video-root /chemin/vers/videos --port 8765
```

`--video-root` délimite les vidéos que le serveur peut lire. Le chemin de la vidéo vient du champ `source.path` dans `run.json`. Il doit exister et se trouver sous ce dossier. Les artefacts du run doivent contenir `run.json`, `observations.jsonl` et `statistics.json`; `tracks.jsonl` fournit les distances progressives.

Les options vidéo sont regroupées par catégorie. **Joueurs** et **Arbitres** ont
chacun un choix **Bbox**, **Segmentation**, **Bbox + segmentation** ou **Masqués**,
ainsi que leurs propres boutons **Pose** et **Étiquettes**. Il est donc possible
de voir seulement les masques des joueurs tout en conservant les bbox arbitres.
Les étiquettes peuvent accompagner les masques sans afficher les boîtes.
Le mode segmentation seule ne dessine aucune bbox de repli si un ancien run n’a
pas de masque. Le décodage RLE et son cache restent bornés.
**Ballon** commande son affichage et sa traînée ; **Terrain** commande ses repères.

La vue terrain montre seulement les personnes dotées d'une `position_m` valide. Le ballon est affiché sur la vidéo avec une traînée décroissante sur les cinq images précédentes, dans la limite du plan courant. Il n'est pas projeté au sol : les artefacts ne contiennent pas sa projection métrique.

Les calques se commandent dans la barre du lecteur, à côté d'« Agrandir la vidéo ». Le choix « Toutes les détections / Personnes identifiées » agit sur les personnes visibles dans la vidéo et sur le terrain ; le second mode ne montre que les joueurs reconnus par le run. Les calques sont synchronisés sur le temps des images vidéo affichées, préchargées par paquets et effacées si l'observation correspondante manque encore. Les joueurs identifiés prennent la couleur de leur équipe dans le roster, ou celle attribuée par le pipeline. La couleur d'équipe des autres joueurs provient du vote majoritaire sur leur torse quand il est concluant.

Quand le navigateur fournit `requestVideoFrameCallback`, son `mediaTime` désigne
l’image présentée et commande les calques, y compris après une recherche en pause.
La destination de recherche ne dessine aucun masque avant cette présentation.
Sans cette API, le studio utilise le dernier timestamp inférieur ou égal au temps
vidéo et attend la fin de recherche. Les masques stockés restent découpés aux bbox
du pipeline ; une silhouette tronquée ou une mauvaise catégorie peut donc venir
des prédictions. Voir l’[audit du décalage à 51,60 s](../reports/segmentation-display-audit-2026-10-25/README.md).

La mise en page tient dans la hauteur de la fenêtre : le lecteur occupe environ deux tiers de la largeur, avec la vue tactique et les effectifs à droite.

Le panneau **Effectifs** permet de choisir l'une des deux équipes, puis un joueur. La fiche du joueur présente sa distance finale `observed_distance_m`, sa durée mesurée et sa couverture, issues de `statistics.json` et indépendantes de la position de lecture. La couverture rapporte la durée mesurée aux intervalles suivis ; elle ne mesure pas la présence réelle hors champ. Les valeurs absentes restent affichées `—`. La flèche revient à la liste de son équipe. Le panneau est conçu pour accueillir d'autres statistiques finales.

Sur les anciens runs dont `statistics.json` omet les numéros, le viewer relit les numéros confirmés dans `observations.jsonl` par track et segment pour les étiquettes vidéo. Si le run a été produit sans effectif mais qu'un `jersey.json` se trouve à côté de la vidéo, cet effectif est montré comme référence d'affichage, sans attribuer de statistiques. Un numéro unique peut alors suggérer une équipe et sa couleur ; la boîte est pointillée car cette association n'a pas été validée par le pipeline. Un numéro porté par les deux équipes reste ambigu.
Le bouton « Agrandir la vidéo » replie la vue terrain et le panneau Effectifs pour agrandir le lecteur.

Raccourcis : espace pour lire/mettre en pause, flèches gauche et droite pour avancer ou reculer d'environ une seconde, Maj + flèches pour une image, `N`/`P` pour les repères suivants/précédents. Le sélecteur de vitesse et le bouton plein écran sont sous la vidéo.

## Vérifier

```bash
python -m unittest viewer.test_server -v
node --check viewer/app.js
node --test viewer/test_app.js
```
