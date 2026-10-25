"""Configure DVC's R2 remote, keeping credentials local and hidden on input."""

import os
import re
import sys
from getpass import getpass
from pathlib import Path
from urllib.parse import urlsplit

from dvc.config import Config


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    if not sys.stdin.isatty():
        raise SystemExit("Lance cette commande dans ton terminal interactif.")

    config = Config()
    existing = config.get("remote", {}).get("r2", {})
    default_bucket = urlsplit(existing.get("url", "")).netloc or "vyneros-dvc"
    bucket = input(f"Nom du bucket R2 [{default_bucket}] : ").strip() or default_bucket
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket):
        raise SystemExit("Nom de bucket R2 invalide.")
    default_endpoint = existing.get("endpointurl", "")
    endpoint = (
        input(f"URL S3 API [{default_endpoint}] : ").strip() or default_endpoint
    ).rstrip("/")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or not (parsed.hostname or "").endswith(".r2.cloudflarestorage.com")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port
        or parsed.path not in ("", f"/{bucket}")
    ):
        raise SystemExit("Utilise l'URL S3 API HTTPS du bucket, pas son URL publique.")
    endpoint = f"https://{parsed.hostname}"

    access_key = getpass("Access Key ID (saisie masquée) : ").strip()
    secret_key = getpass("Secret Access Key (saisie masquée) : ").strip()
    if not access_key or not secret_key:
        raise SystemExit("Les deux clés sont nécessaires ; configuration inchangée.")

    with config.edit() as shared:
        shared.setdefault("core", {})["remote"] = "r2"
        shared.setdefault("remote", {})["r2"] = {
            "url": f"s3://{bucket}/vyneros",
            "endpointurl": endpoint,
            "region": "auto",
        }

    previous_umask = os.umask(0o077)
    try:
        with config.edit(level="local") as local:
            local.setdefault("remote", {})["r2"] = {
                "access_key_id": access_key,
                "secret_access_key": secret_key,
            }
        Path(".dvc/config.local").chmod(0o600)
    finally:
        os.umask(previous_umask)

    print("R2 configuré. Clés enregistrées uniquement dans .dvc/config.local.")
    print("Aucune donnée envoyée. Lance make data-push pour transférer le dataset.")


if __name__ == "__main__":
    main()
