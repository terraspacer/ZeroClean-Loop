"""Freeze the preregistered v3 maximum-objective sensitivity before dirty fits."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path

from zeroclean.data import DATASETS, load_dirty
from experimental import latent_consensus as v1
from experimental import latent_consensus_v2 as v2
from experimental import latent_consensus_v3 as experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--entity", required=True, nargs="+")
    parser.add_argument("--context", default=[], nargs="*")
    parser.add_argument("--source", required=True)
    parser.add_argument("--attributes", required=True, nargs="+")
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parameters = experiment.SkillParameters()
    table = load_dirty(args.dataset)
    sources = {
        "model_frozen.py": Path(experiment.__file__).read_bytes(),
        "v1_helpers_frozen.py": Path(v1.__file__).read_bytes(),
        "v2_helpers_frozen.py": Path(v2.__file__).read_bytes(),
        "protocol_frozen.md": Path(experiment.__file__).with_name("latent_consensus_v3_protocol.md").read_bytes(),
        "runner_frozen.py": Path(__file__).read_bytes(),
    }
    manifest = {"experiment": experiment.VERSION, "dataset": table.name,
                "dirty_fingerprint": table.fingerprint,
                "hypothetical": True, "hypothesis": args.hypothesis,
                "entity_columns": args.entity, "context_columns": args.context,
                "source_column": args.source, "attributes": args.attributes,
                "parameters": asdict(parameters),
                "compared_source_assumptions": ["independent_names", "cluster_near_agreement"],
                "frozen_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in sources.items()},
                "promotion": "Never applied to the live model or dirty table",
                "interpretation": "Orientation/source-assumption stability, not empirical correctness"}
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, payload in sources.items():
        with (args.output / filename).open("xb") as stream:
            stream.write(payload)
    with (args.output / "pre_fit_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    summary = {"manifest": manifest, "attributes": []}
    supported = {"propose", "consistent"}
    for index, attribute in enumerate(args.attributes):
        contract = experiment.SkillContract(
            entity_columns=tuple(args.entity), context_columns=tuple(args.context),
            source_column=args.source, value_column=attribute,
            single_valued=True, independent_sources=True, stationary_reliability=True,
            ignorable_missingness=True, orientation_prior_disclosed=True,
            evidence=args.hypothesis, hypothetical=True)
        item = {"attribute": attribute, "source_assumptions": {}}
        results = {}
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
                "fold_blockers": [{"fold": f["fold"], "blockers": f["blockers"]} for f in result["folds"]],
                "supported_entities": len({tuple(d["entity"]) for d in result["decisions"] if d["status"] in supported}),
                "sign_ambiguous_sources_by_fold": {str(f["fold"]): f.get("sign_ambiguous_sources", []) for f in result["folds"]}}
            print(json.dumps({"attribute": attribute, "mode": mode,
                              "status_counts": result["status_counts"],
                              "fold_blockers": item["source_assumptions"][mode]["fold_blockers"]}), flush=True)
        a, b = (results[m]["decisions"] for m in manifest["compared_source_assumptions"])
        common = [(x, y) for x, y in zip(a, b) if x["status"] in supported and y["status"] in supported]
        item["both_assumptions_supported_rows"] = len(common)
        item["supported_winner_disagreement_rows"] = sum(x["candidate"] != y["candidate"] for x, y in common)
        item["stable_proposal_previews"] = [
            {"row": x["row"], "column": attribute, "entity": x["entity"], "source": x["source"],
             "original": x["original"], "candidate": x["candidate"]}
            for x, y in common if x["status"] == y["status"] == "propose" and x["candidate"] == y["candidate"]]
        item["stable_proposal_rows"] = len(item["stable_proposal_previews"])
        item["stable_proposal_entities"] = len({tuple(d["entity"]) for d in item["stable_proposal_previews"]})
        summary["attributes"].append(item)
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
