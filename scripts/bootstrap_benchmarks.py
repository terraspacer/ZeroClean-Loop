#!/usr/bin/env python3
"""Clone pinned ZeroED/ZeroEC benchmark checkouts and optionally unpack the v1.0.0 release artifact bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_checkout(repo_url: str, commit: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").exists():
        subprocess.run(["git", "clone", repo_url, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", commit], check=True)


def verify_csvs(expected: dict[str, str]) -> None:
    for rel, digest in sorted(expected.items()):
        path = ROOT / rel
        if not path.is_file():
            raise SystemExit(f"Missing benchmark CSV: {rel}")
        actual = sha256(path)
        if actual != digest:
            raise SystemExit(f"SHA-256 mismatch for {rel}: expected {digest}, got {actual}")
    print(f"Verified {len(expected)} benchmark CSV files under vendor/ZeroED/data/.")


def unpack_bundle(archive_path: Path, expected_sha256: str) -> None:
    actual = sha256(archive_path)
    if actual != expected_sha256:
        raise SystemExit(f"Release bundle SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            dest = (ROOT / member.name).resolve()
            if not str(dest).startswith(str(ROOT.resolve()) + "/"):
                raise SystemExit(f"Unsafe path in archive: {member.name}")
        tf.extractall(ROOT)
    print(f"Verified and unpacked {archive_path.name} (sha256={actual[:12]}...).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-release-bundle", action="store_true",
                        help="Download and unpack the SHA-256-verified v1.0.0 audit/input bundle from GitHub Releases.")
    parser.add_argument("--bundle-file", type=Path,
                        help="Unpack a pre-downloaded zeroclean-loop-v6-audit-and-inputs.tar.gz file.")
    args = parser.parse_args()

    lock = json.loads((ROOT / "third_party.lock.json").read_text(encoding="utf-8"))
    for key in ("zeroed", "zeroec"):
        spec = lock[key]
        ensure_checkout(spec["repository"], spec["commit"], ROOT / spec["checkout_path"])
    verify_csvs(lock["zeroed"]["benchmark_csv_sha256"])

    bundle = lock["release_bundle_v1"]
    if args.bundle_file:
        unpack_bundle(args.bundle_file, bundle["sha256"])
    elif args.download_release_bundle:
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / bundle["filename"]
            print(f"Downloading {bundle['url']} ...")
            urllib.request.urlretrieve(bundle["url"], dest)
            unpack_bundle(dest, bundle["sha256"])


if __name__ == "__main__":
    main()
