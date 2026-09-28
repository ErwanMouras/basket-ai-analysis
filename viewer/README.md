# Studio d'analyse vidéo

Visualiseur local en lecture seule pour les runs terminés du pipeline. Il affiche la vidéo source, les calques de détection (joueurs, arbitres, ballon, pose, identifiants, repères terrain), une vue du terrain projeté et la distance observable cumulée à l'instant de lecture.

## Démarrer

Depuis la racine du dépôt :

```bash
python -m viewer.server
```

Ouvrir <http://127.0.0.1:8765>. Le sélecteur recense les runs terminés présents sous `runs/analysis`. Le run `validation/color-groups` s'ouvre par défaut s'il existe, puis `validation/optimized-roster` à défaut.

Pour utiliser d'autres dossiers :

```bash
python -m viewer.server --runs-root /chemin/vers/runs --video-root /chemin/vers/videos --port 8765
```

`--video-root` délimite les vidéos que le serveur peut lire. Le chemin de la vidéo vient du champ `source.path` dans `run.json`. Il doit exister et se trouver sous ce dossier. Les artefacts du run doivent contenir `run.json`, `observations.jsonl` et `statistics.json`; `tracks.jsonl` fournit les distances progressives.

La vue terrain montre seulement les personnes dotées d'une `position_m` valide. Le ballon est affiché sur la vidéo, mais pas au sol : les artefacts ne contiennent pas sa projection métrique. La somme progressive additionne les contributions `distance_m` jusqu'à l'image courante. Elle peut inclure plusieurs tracks anonymes d'une même personne et ne représente pas la distance totale du match.

Raccourcis : espace pour lire/mettre en pause, flèches gauche et droite pour avancer ou reculer d'une seconde. Le sélecteur de vitesse et le bouton plein écran sont sous la vidéo.

## Vérifier

```bash
python -m unittest viewer.test_server -v
node --check viewer/app.js
```
