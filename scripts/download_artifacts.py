"""Telecharge les artefacts publics verifies, sans ecraser une destination existante."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import urllib.request
import zipfile


def extract_verified(archive, destination):
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 1000 or sum(e.file_size for e in entries) > 200_000_000:
            raise ValueError("Archive trop volumineuse")
        for entry in entries:
            target = (root / entry.filename).resolve()
            if not target.is_relative_to(root) or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Chemin d'archive non autorise")
        for entry in entries:
            target = (root / entry.filename).resolve()
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("artifacts/v2"))
    args = parser.parse_args()
    dest = args.dest.resolve()
    if dest.exists():
        parser.error("Destination existante : choisir un autre --dest")
    manifest = json.loads((Path(__file__).resolve().parents[1] / "docs/artifacts-v2.json").read_text())
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nasse-download-", dir=dest.parent) as temporary:
        tmp = Path(temporary)
        archive = tmp / "bundle.zip"
        digest, total = hashlib.sha256(), 0
        request = urllib.request.Request(manifest["url"], headers={"User-Agent": "nasse-sim-artifacts"})
        with urllib.request.urlopen(request, timeout=60) as source, archive.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > manifest["size"]:
                    raise ValueError("Taille inattendue")
                digest.update(chunk)
                output.write(chunk)
        if total != manifest["size"] or digest.hexdigest() != manifest["sha256"]:
            raise ValueError("Verification SHA256/taille echouee")
        extracted = tmp / "extracted"
        extracted.mkdir()
        extract_verified(archive, extracted)
        if dest.exists():
            raise FileExistsError(dest)
        extracted.rename(dest)
    print(f"Artefacts verifies et extraits : {dest}")


if __name__ == "__main__":
    main()
