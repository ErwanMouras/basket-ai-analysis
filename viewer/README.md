# Studio d'analyse vidéo

Visualiseur local en lecture seule pour les runs terminés du pipeline. Il affiche la vidéo source, les calques de détection (joueurs, arbitres, ballon, pose, identifiants, repères terrain), une vue du terrain projeté et la distance observable cumulée à l'instant de lecture.

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

La vue terrain montre seulement les personnes dotées d'une `position_m` valide. Le ballon est affiché sur la vidéo, mais pas au sol : les artefacts ne contiennent pas sa projection métrique. La somme progressive additionne les contributions `distance_m` jusqu'à l'image courante. Les tracks dont l'identité est résolue sont regroupées par joueur, dès leur première image ; les tracks anonymes peuvent encore représenter une même personne. La table expose les joueurs identifiés et les tracks anonymes, avec recherche, filtre, couverture finale et détail des exclusions.

Les calques se commandent dans la barre du lecteur, à côté d'« Agrandir la vidéo ». Les boîtes sont synchronisées sur le temps des images vidéo affichées, préchargées par paquets et effacées si l'observation correspondante manque encore. Les arbitres gardent une couleur constante. Les joueurs identifiés prennent la couleur de leur équipe dans le roster, ou celle attribuée par le pipeline. La couleur d'équipe des autres joueurs provient du vote majoritaire sur leur torse quand il est concluant. Le panneau d'inspection affiche les totaux des votes pour le numéro et l'équipe, ainsi que les lectures OCR numériques retenues et rejetées. Cliquer une lecture va à l'image source et affiche son crop depuis la vidéo.

Sur les anciens runs dont `statistics.json` omet les numéros, le viewer relit les numéros confirmés dans `observations.jsonl` par track et segment. Un `*` sur une boîte indique que le numéro a été confirmé ailleurs sur la même track, pas sur l'image courante. Si le run a été produit sans effectif mais qu'un `jersey.json` se trouve à côté de la vidéo, cet effectif est montré comme **référence d'affichage**. Un numéro unique peut alors suggérer une équipe et sa couleur ; la boîte est pointillée et le panneau d'inspection précise que cette association n'a pas été validée par le pipeline. Un numéro porté par les deux équipes reste ambigu.
Le bouton « Agrandir la vidéo » replie les panneaux secondaires pour inspecter les boîtes à plus grande taille.

Raccourcis : espace pour lire/mettre en pause, flèches gauche et droite pour avancer ou reculer d'environ une seconde, Maj + flèches pour une image, `N`/`P` pour les repères suivants/précédents. Le sélecteur de vitesse et le bouton plein écran sont sous la vidéo.

## Vérifier

```bash
python -m unittest viewer.test_server -v
node --check viewer/app.js
```
