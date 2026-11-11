"""Inference types intentionally have no reference-table field or loader."""
from __future__ import annotations
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

DATASETS = ("hospital", "flights", "beers", "rayyan", "billionaire", "movies")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class DirtyTable:
    name: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    fingerprint: str

    @property
    def shape(self):
        return len(self.rows), len(self.columns)

    def records(self):
        return [dict(zip(self.columns, row)) for row in self.rows]


def read_csv(path: str | Path):
    """Preserve all strings, empty cells, row order, and leading zeros exactly."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        columns = tuple(next(reader))
        rows = tuple(tuple(row) for row in reader)
    if len(set(columns)) != len(columns):
        raise ValueError(f"Duplicate columns in {path}")
    if any(len(row) != len(columns) for row in rows):
        raise ValueError(f"Ragged table: {path}")
    return columns, rows


def load_dirty(name: str, root: str | Path = "vendor/ZeroED/data") -> DirtyTable:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    path = Path(root) / f"{name}_error-01.csv"
    columns, rows = read_csv(path)
    return DirtyTable(name, columns, rows, sha256(path))
