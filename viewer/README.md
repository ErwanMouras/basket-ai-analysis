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

Le calque **Segmentation** superpose les masques des joueurs et arbitres lorsqu’ils
sont présents dans le run. Il se coche indépendamment de la pose et des étiquettes,
et respecte les filtres de personnes. Les anciens runs sans masques restent
compatibles. Le décodage RLE reste limité aux régions utiles et son cache est borné.

La vue terrain montre seulement les personnes dotées d'une `position_m` valide. Le ballon est affiché sur la vidéo avec une traînée décroissante sur les cinq images précédentes, dans la limite du plan courant. Il n'est pas projeté au sol : les artefacts ne contiennent pas sa projection métrique.

Les calques se commandent dans la barre du lecteur, à côté d'« Agrandir la vidéo ». Le choix « Toutes les détections / Personnes identifiées » agit sur les boîtes et les positions du terrain ; le second mode ne montre que les joueurs reconnus par le run. Les boîtes sont synchronisées sur le temps des images vidéo affichées, préchargées par paquets et effacées si l'observation correspondante manque encore. Les joueurs identifiés prennent la couleur de leur équipe dans le roster, ou celle attribuée par le pipeline. La couleur d'équipe des autres joueurs provient du vote majoritaire sur leur torse quand il est concluant.

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
