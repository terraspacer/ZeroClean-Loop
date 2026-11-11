"""Freeze and run an explicitly hypothetical, unpromoted dirty-only preview.

The two source assumptions and all thresholds are fixed before fitting. No
reference tables, evaluations, labels, judgments, or baseline outputs are read.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path

from zeroclean.data import DATASETS, load_dirty
from experimental import latent_consensus as experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--entity", required=True, nargs="+")
    parser.add_argument("--context", default=[], nargs="*")
    parser.add_argument("--source", required=True)
    parser.add_argument("--attributes", required=True, nargs="+")
    parser.add_argument("--hypothesis", required=True,
                        help="Explicitly disclosed semantic and provenance hypotheses")
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    args = parser.parse_args()
    parameters = experiment.Parameters()
    table = load_dirty(args.dataset)
    module_path = Path(experiment.__file__)
    code = module_path.read_bytes()
    protocol = module_path.with_name("latent_consensus_protocol.md").read_bytes()
    runner = Path(__file__).read_bytes()
    manifest = {
        "experiment": experiment.VERSION, "dataset": table.name,
        "dirty_fingerprint": table.fingerprint,
        "hypothesis": args.hypothesis, "hypothetical": True,
        "entity_columns": args.entity, "context_columns": args.context,
        "source_column": args.source, "attributes": args.attributes,
        "parameters": asdict(parameters),
        "compared_source_assumptions": ["independent_names", "cluster_near_agreement"],
        "code_sha256": hashlib.sha256(code).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol).hexdigest(),
        "runner_sha256": hashlib.sha256(runner).hexdigest(),
        "promotion": "Never applied to the live model or dirty table",
        "interpretation": "Assumption-bound model stability; no accuracy or calibration claim",
    }
    # Exclusive directory/file creation also freezes the source and protocol
    # before fitting any dirty-table model.
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, payload in (("model_frozen.py", code), ("protocol_frozen.md", protocol),
                              ("runner_frozen.py", runner)):
        with (args.output / filename).open("xb") as stream:
            stream.write(payload)
    with (args.output / "pre_fit_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    summary = {"manifest": manifest, "attributes": []}
    supported = {"propose", "consistent"}
    for index, attribute in enumerate(args.attributes):
        contract = experiment.Contract(
            entity_columns=tuple(args.entity), context_columns=tuple(args.context),
            source_column=args.source, value_column=attribute,
            single_valued=True, independent_sources=True, stationary_reliability=True,
            better_than_half=True, ignorable_missingness=True,
            evidence=args.hypothesis, hypothetical=True)
        results = {}
        item = {"attribute": attribute, "source_assumptions": {}}
        for mode in manifest["compared_source_assumptions"]:
            result = experiment.run_experiment(table, contract, replace(parameters, source_assumption=mode))
            results[mode] = result
            filename = f"attribute_{index}_{mode}.json.gz"
            with gzip.open(args.output / filename, "xt", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, separators=(",", ":"))
                stream.write("\n")
            item["source_assumptions"][mode] = {
                "artifact": filename, "fit_performed": result["fit_performed"],
                "status_counts": result["status_counts"],
                "fold_blockers": [{"fold": f["fold"], "blockers": f["blockers"]}
                                  for f in result["folds"]],
                "supported_entities": len({tuple(d["entity"]) for d in result["decisions"]
                                           if d["status"] in supported}),
            }
        a, b = (results[m]["decisions"] for m in manifest["compared_source_assumptions"])
        common = [(x, y) for x, y in zip(a, b)
                  if x["status"] in supported and y["status"] in supported]
        item["both_assumptions_supported_rows"] = len(common)
        item["supported_winner_disagreement_rows"] = sum(x["candidate"] != y["candidate"] for x, y in common)
        item["stable_proposal_previews"] = [
            {"row": x["row"], "column": attribute, "entity": x["entity"],
             "source": x["source"], "original": x["original"], "candidate": x["candidate"]}
            for x, y in common if x["status"] == y["status"] == "propose" and x["candidate"] == y["candidate"]]
        item["stable_proposal_rows"] = len(item["stable_proposal_previews"])
        item["stable_proposal_entities"] = len({tuple(d["entity"]) for d in item["stable_proposal_previews"]})
        summary["attributes"].append(item)
        print(json.dumps({"attribute": attribute,
                          "source_assumptions": item["source_assumptions"],
                          "stable_proposal_rows": item["stable_proposal_rows"],
                          "stable_proposal_entities": item["stable_proposal_entities"]}))
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
