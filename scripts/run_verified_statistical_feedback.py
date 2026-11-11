#!/usr/bin/env python3
"""Prepare or apply standalone verified statistical feedback; never queue/evaluate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experimental.verified_statistical_feedback import prepare_from_files, apply_from_files
from zeroclean.data import DATASETS, load_dirty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    prepare = stages.add_parser("prepare", help="Freeze a NEW verifier request from explicit original inputs")
    apply = stages.add_parser("apply", help="Consume complete independent judgments into a NEW output root")
    for command in (prepare, apply):
        command.add_argument("--dataset", choices=DATASETS, required=True)
        command.add_argument("--output", type=Path, required=True)
    for name in ("bundle", "freeze-manifest", "prior", "frozen-v3", "artifact"):
        prepare.add_argument("--" + name, type=Path, required=True)
    apply.add_argument("--request", type=Path, required=True)
    apply.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    table = load_dirty(args.dataset)
    if args.stage == "prepare":
        result = prepare_from_files(table, args.bundle, args.freeze_manifest, args.prior, args.frozen_v3, args.artifact, args.output)
        print(json.dumps({"output": str(args.output), "request_hash": result["request_hash"], "cells": len(result["cells"])}, indent=2))
    else:
        result = apply_from_files(table, args.request, args.verification, args.output)
        print(json.dumps({"output": str(args.output), "prediction_files": result["prediction_files"], "export_hash": result["export_hash"]}, indent=2))


if __name__ == "__main__":
    main()
