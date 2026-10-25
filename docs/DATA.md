# Données : DVC et Cloudflare R2

`datas.dvc` référence le contenu de `datas/` : vidéos, annotations et métadonnées.
Les dossiers `nba_cache/` sont exclus par `.dvcignore` et restent locaux.
Les données sont ignorées par Git ; les références DVC sont à committer.
Le dataset initial contient 43 fichiers pour environ 421 Mo hors caches.

Les annotations joueurs locales `*.playersann.json` sont également couvertes par
`datas.dvc` lorsqu'elles se trouvent sous `datas/`. Les verrous
`*.playersann.json.lock` sont exclus par `.dvcignore`. Après annotation, fermer
l'éditeur puis utiliser `.venv/bin/dvc status` et `.venv/bin/dvc add datas` pour
actualiser **localement** le cache et la référence. Examiner puis committer
`datas.dvc` séparément ; aucun envoi n'est nécessaire à l'export.

`make export-players` produit les deux formats régénérables YOLO/COCO dans
`exports/players/`, hors de `datas.dvc`. Cette commande ne lance aucun transfert
DVC ni enregistrement automatique des sources. `make verify-players-export`
contrôle la paire publiée sans les médias sources. Voir le
[guide joueurs](../training/players/README.md#exporter-les-annotations-locales-en-yolo-et-coco).

Les sources sont maintenant rangées sous `datas/train/` et `datas/val/`.
Ce premier dossier définit le split pour l'export balle ; le champ `split` d'un
éventuel `.meta.yaml` doit être cohérent. Après déplacement ou correction des
métadonnées, `.venv/bin/dvc add datas` enregistre ce nouvel état localement.

Les datasets générés par `make export-ball` se trouvent dans `exports/ball/` et sont
régénérables depuis les sources/configurations. Ils sont ignorés par Git, hors
du suivi `datas.dvc`, et ne sont donc pas envoyés par `.venv/bin/dvc push`.
Voir le [configuration d'export](../training/ball/configs/ball_export.yaml) pour leurs manifestes et le
suivi MLflow actuel.

## Installation

Depuis la racine du dépôt, avec l'environnement Python du projet :

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

La dépendance `dvc[s3]==3.67.1` inclut le connecteur compatible R2.
Le Makefile utilise DVC dans `.venv/`. Pour une autre installation, passer
`DVC=/chemin/vers/dvc` et `PYTHON=/chemin/vers/python` selon la commande.
Il n'est pas nécessaire de refaire `dvc init` après un clone.

## Créer le stockage dans Cloudflare

1. Dans le dashboard, ouvrir **Storage & databases > R2 > Overview**.
   Activer R2 si nécessaire. Le quota gratuit ne bloque pas les dépassements
   facturables ; consulter les tarifs affichés à l'activation.
2. Créer un bucket, par exemple `vyneros-dvc`, en classe **Standard**.
   Conserver l'accès public désactivé.
3. Dans la gestion des **R2 API tokens**, créer un token avec la permission
   **Object Read & Write**, limitée à ce bucket.
4. Conserver l'**Access Key ID**, la **Secret Access Key** et l'URL **S3 API**.
   Le token API Cloudflare générique ne remplace pas cette paire de clés S3.

L'URL S3 ressemble à `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.
Certains buckets utilisent une URL de juridiction, notamment avec `.eu.` :
copier celle fournie par Cloudflare. Le script accepte aussi le suffixe
`/nom-du-bucket` affiché dans les paramètres et le retire de l'endpoint.

## Configurer la machine

Dans un terminal interactif :

```bash
.venv/bin/python scripts/configure_r2.py
```

Saisir le bucket, l'URL S3 et les deux clés. Les clés sont masquées pendant la
saisie et enregistrées dans `.dvc/config.local`, ignoré par Git, avec des
permissions de lecture/écriture réservées à l'utilisateur sur Linux.
Ne pas les coller dans une conversation ou les committer.

Le script définit `r2` comme remote par défaut, sous `s3://<bucket>/vyneros`.
Seuls le bucket, l'endpoint et la région `auto` sont écrits dans `.dvc/config`,
qui est partageable. Aucun transfert n'est effectué par la configuration.

## Premier envoi et vérification

```bash
.venv/bin/dvc status
.venv/bin/dvc push
.venv/bin/dvc status --cloud
```

`.venv/bin/dvc push` envoie les contenus déjà enregistrés par DVC. Si les fichiers
ont changé depuis le dernier enregistrement, fermer l'annotateur et lancer
`.venv/bin/dvc add datas` avant l'envoi.

Dans R2, les objets sont rangés sous `vyneros/files/md5/` avec des noms calculés
à partir de leur contenu. C'est normal : DVC reconstitue les noms d'origine.
Ne pas déplacer manuellement ces objets ou leur appliquer une expiration
automatique, car d'anciens commits peuvent encore les référencer.

Après un transfert réussi, enregistrer les fichiers de configuration et les
références avec les modifications de code correspondantes :

```bash
git add .dvc/config .dvc/.gitignore .dvcignore .gitignore datas.dvc
git add requirements.txt Makefile scripts/configure_r2.py docs/DATA.md README.md
git diff --cached --stat
git commit -m "Track datasets with DVC and configure Cloudflare R2"
```

Le code de l'annotateur doit lui aussi être enregistré dans Git pour permettre
sa récupération sur une autre machine. Un `git push` envoie les commits à
GitHub, tandis que `.venv/bin/dvc push` envoie les données à R2.

## Après une session d'annotation

```bash
.venv/bin/dvc status
.venv/bin/dvc add datas
.venv/bin/dvc push
git add datas.dvc
git commit -m "Update ball annotations"
```

Les vidéos inchangées sont réutilisées. Les nouvelles versions des fichiers
modifiés occupent du stockage supplémentaire. Les anciennes versions restent
accessibles tant que leur contenu est conservé dans R2 ou le cache local.

## Récupérer les données sur une autre machine

Cloner le dépôt, installer ses dépendances puis configurer les clés locales
avec `.venv/bin/python scripts/configure_r2.py`, en reprenant le même bucket et le même endpoint.
Pour un utilisateur qui télécharge seulement, créer un token R2 limité à
**Object Read only** sur ce bucket.

```bash
.venv/bin/dvc pull
.venv/bin/dvc status
```

`data-pull` restaure la version référencée par le commit Git courant. Avant de
changer de commit ou de récupérer une autre version, enregistrer son travail
d'annotation. Avec un bucket privé, un visiteur GitHub a besoin de clés
autorisées : la publication du code ne donne pas accès aux données.

## Documentation officielle

- [Activer R2](https://developers.cloudflare.com/r2/get-started/)
- [Créer des clés R2](https://developers.cloudflare.com/r2/api/tokens/)
- [Configurer un remote S3 compatible dans DVC](https://doc.dvc.org/user-guide/data-management/remote-storage/amazon-s3)
- [Tarifs R2](https://developers.cloudflare.com/r2/pricing/)
