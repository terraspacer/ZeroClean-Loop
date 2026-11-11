#!/usr/bin/env python3
"""Start/resume external verifier feedback; output remains separate until promoted."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zeroclean.closed_loop import advance_closed_loop, start_closed_loop
from zeroclean.data import load_dirty, sha256
from zeroclean.llm import FileLLMQueue
from zeroclean.types import digest


def source_signature():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "zeroclean").glob("*.py")) + [Path(__file__).resolve()]
    return digest({str(path.relative_to(root)): sha256(path) for path in paths})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("start", "resume"), required=True)
    parser.add_argument("--payload", type=Path, help="Hash-bound initial candidate request")
    parser.add_argument("--verification", type=Path, help="Complete response to the initial request")
    parser.add_argument("--artifact", type=Path, help="Exact pass2 artifact used by the initial request")
    parser.add_argument("--output", type=Path, default=Path("artifacts/closed_loop"))
    parser.add_argument("--run-dir", type=Path, help="Exact versioned run directory to resume")
    parser.add_argument("--max-rounds", type=int, default=3, help="External round limit fixed when starting a new run")
    parser.add_argument("--queue", action="store_true", help="Queue fresh missing verifier judgments")
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    signature = source_signature()
    if args.stage == "start":
        if any(value is None for value in (args.payload, args.verification, args.artifact)):
            parser.error("start requires --payload, --verification, and --artifact")
        payload = json.loads(args.payload.read_text())
        table = load_dirty(payload["dataset"])
        directory = start_closed_loop(table, args.payload, args.verification, args.artifact,
                                      args.output, max_rounds=args.max_rounds, source_signature=signature)
    else:
        if args.run_dir is None:
            parser.error("resume requires --run-dir")
        manifest = json.loads((args.run_dir / "manifest.json").read_text())
        table = load_dirty(manifest["identity"]["dataset"])
        directory = args.run_dir
    callback = FileLLMQueue().request if args.queue else None
    with (directory / ".resume.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status = advance_closed_loop(table, directory, source_signature=signature,
                                     request=callback, batch_size=args.batch_size)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
