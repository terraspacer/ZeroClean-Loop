#!/usr/bin/env python3
"""Sequentially prepare source-equivalent fastText dimensions for ZeroED."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeroclean.fasttext_resources import DEFAULT_DIMENSIONS, prepare_fasttext_dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("artifacts/models/cc.en.300.bin"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/reduced"))
    parser.add_argument("--dimensions", type=int, nargs="+", default=DEFAULT_DIMENSIONS)
    args = parser.parse_args()
    result = prepare_fasttext_dimensions(args.original, args.output, args.dimensions,
                                         progress=lambda item: print(json.dumps(item), flush=True))
    print(json.dumps({"stage": "complete", "dimensions": result["dimensions"]}), flush=True)


if __name__ == "__main__":
    main()
