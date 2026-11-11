from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd,name=tempfile.mkstemp(prefix="."+path.name+".",suffix=".tmp",dir=path.parent)
    temp=Path(name)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass
class Prediction:
    probabilities: np.ndarray
    repairs: dict[tuple[int, int], str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def validate(self, shape):
        self.probabilities = np.asarray(self.probabilities, dtype=float)
        if self.probabilities.shape != shape:
            raise ValueError("Prediction shape mismatch")
        if not np.isfinite(self.probabilities).all() or ((self.probabilities < 0) | (self.probabilities > 1)).any():
            raise ValueError("Probabilities must be finite and in [0,1]")
        for (i, j), value in self.repairs.items():
            if not 0 <= i < shape[0] or not 0 <= j < shape[1] or not isinstance(value, str):
                raise ValueError("Invalid repair coordinate/value")
        return self

    def save(self, path):
        body = {"probabilities": self.probabilities.tolist(),
                "repairs": [{"row": i, "column": j, "value": v} for (i,j),v in sorted(self.repairs.items())],
                "metadata": self.metadata}
        write_json(path, body)
        return digest(body)

    @classmethod
    def load(cls, path):
        body = json.loads(Path(path).read_text())
        return cls(np.asarray(body["probabilities"]),
                   {(r["row"], r["column"]): r["value"] for r in body["repairs"]}, body["metadata"])
