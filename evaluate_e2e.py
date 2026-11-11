#!/usr/bin/env python3
"""Reproducible stages: prepare dirty-only requests, run/freeze, then evaluate."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from zeroclean.data import DATASETS, load_dirty, sha256
from zeroclean.types import Prediction, digest, write_json
from zeroclean.llm import load_artifact
from zeroclean.baselines import adapted_zeroed, adapted_zeroed_zeroec, preflight_zeroed, preflight_zeroec
from zeroclean.loop import LoopConfig, build_candidates, infer_from_candidates
from zeroclean.optimize import hill_climb


def source_signature():
    paths=sorted(Path("zeroclean").glob("*.py"))
    return digest({str(p):sha256(p) for p in paths})


def matched_source_signature():
    return digest({"inference_sources":source_signature(),"runner":sha256(Path(__file__))})


def prepare_ablations(table,out,config,detector="adapted"):
    """Prepare a separate matched experiment without changing legacy artifacts."""
    from zeroclean.ablations import prepare_matched_ablations
    baseline_name="zeroed" if detector=="source" else "adapted_zeroed"
    destination=prepare_matched_ablations(
        table,out,Path("artifacts/llm")/table.name/"pass2.json",
        out/"predictions"/table.name/f"{baseline_name}.json",detector=detector,
        config=config,source_signature=matched_source_signature())
    print(f"[{table.name}] prepared independent matched-ablation requests: {destination}",flush=True)
    return destination


def model_metadata(metadata):
    """Keep provenance inspectable without embedding a whole detection mask."""
    result={k:v for k,v in metadata.items() if k not in ("trace","columns","upstream_binary_decisions")}
    if "upstream_binary_decisions" in metadata:
        decisions=metadata["upstream_binary_decisions"]
        mask=np.asarray(decisions,dtype=bool)
        result["upstream_detection_mask"]={"sha256":digest(decisions),"shape":list(mask.shape),
                                            "positive_cells":int(mask.sum())}
    return result


def comparison_summary(metrics,assess_targets):
    """Keep qualified reconstruction claims separate from the exact-baseline gate."""
    official=assess_targets(metrics)
    qualified=assess_targets(metrics,"zeroed","zeroed_zeroec_reconstruction")
    if official.get("status")=="verified": status="measured_experiment_exact_baselines_present"
    elif any("zeroed_zeroec_reconstruction" in variants for variants in metrics.values()):
        status="measured_experiment_qualified_source_reconstruction"
    elif any("adapted_zeroed" in variants for variants in metrics.values()):
        status="measured_experiment_adapted_comparison_only"
    else: status="measured_experiment_incomplete_baselines"
    return {"status":status,"official_target_assessment":official,
            "adapted_comparison_only":assess_targets(metrics,"adapted_zeroed","adapted_zeroed_zeroec"),
            "qualified_source_reconstruction_comparison":{
                **qualified,"official_exact_baseline":False,"fixed_example_count":0,
                "fidelity_caveat":"Author-recovered ZeroEC system prompt; historical identity unproven; zero fixed examples."}}


def preflight(out):
    model=Path("artifacts/models/cc.en.300.bin")
    body={"zeroed":preflight_zeroed(fasttext_path=model if model.exists() else None),"zeroec":preflight_zeroec(),"sources":{},
          "python":platform.python_version(),"experiment_llm":{"model":"gpt-6.1-sol","reasoning_effort":"xhigh"}}
    for name in ("ZeroED","ZeroEC"):
        result=subprocess.run(["git","-C",f"vendor/{name}","rev-parse","HEAD"],capture_output=True,text=True)
        body["sources"][name]={"url":f"https://github.com/{'WelkinNi' if name=='ZeroED' else 'YangChen32768'}/{name}",
                              "commit":result.stdout.strip() if result.returncode==0 else None}
    write_json(out/"baseline_preflight.json",body)
    return body


def prepare(table,out,config,force=False,detector="adapted"):
    pass1,pass2=load_artifact(table,1),load_artifact(table,2)
    signature=digest({"dirty":table.fingerprint,"pass1":pass1,"source":source_signature()})
    baseline_path=out/"predictions"/table.name/"adapted_zeroed.json"
    if baseline_path.exists() and not force:
        baseline=Prediction.load(baseline_path)
        if baseline.metadata.get("cache_signature")!=signature: baseline=None
    else: baseline=None
    if baseline is None:
        print(f"[{table.name}] fitting adapted single-pass detector",flush=True)
        baseline=adapted_zeroed(table,pass1)
        baseline.metadata["cache_signature"]=signature
        baseline.save(baseline_path)
    # Independently rebuild the cheap sequential baseline even when ED is cached.
    adapted_zeroed_zeroec(table,baseline,pass1).save(out/"predictions"/table.name/"adapted_zeroed_zeroec.json")
    print(f"[{table.name}] validating rules and preparing counterfactual candidates",flush=True)
    detector_prediction=baseline
    if detector=="source":
        detector_prediction=Prediction.load(out/"predictions"/table.name/"zeroed.json")
        if detector_prediction.metadata.get("reproduced_upstream") is False:
            raise ValueError("Source detector requested but only adapted predictions are available")
    payload=build_candidates(table,pass2,detector_prediction,config)
    payload["detector_provenance"]={"kind":detector,"prediction_hash":digest({
        "probabilities":detector_prediction.probabilities.tolist(),"metadata":detector_prediction.metadata})}
    payload["request_hash"]=digest({k:v for k,v in payload.items() if k!="request_hash"})
    write_json(Path("artifacts/candidates")/f"{table.name}.json",payload)
    write_json(out/"rule_validation"/f"{table.name}.json",payload["rule_validation"])
    return baseline,payload


def run(table,out,config,optimize=False):
    candidate_path=Path("artifacts/candidates")/f"{table.name}.json"
    payload=json.loads(candidate_path.read_text())
    pass2=load_artifact(table,2)
    if payload["artifact_hash"]!=digest(pass2): raise ValueError("Rules changed; rerun prepare")
    verification_path=Path("artifacts/verifications")/f"{table.name}.json"
    verification=json.loads(verification_path.read_text()) if verification_path.exists() else None
    inputs={str(candidate_path):sha256(candidate_path)}
    if verification is not None: inputs[str(verification_path)]=sha256(verification_path)
    if optimize:
        config,prediction,search=hill_climb(table,payload,verification,config)
        write_json(out/"optimization"/f"{table.name}.json",search)
    else: prediction=infer_from_candidates(table,payload,verification,config)
    dest=out/"predictions"/table.name
    prediction.save(dest/"zeroclean_loop.json")
    for name,variant in (("without_counterfactual",replace(config,counterfactual=False)),
                         ("without_iterative_verification",replace(config,iterative_verification=False))):
        result=infer_from_candidates(table,payload,verification,variant)
        result.metadata["ablation_protocol"]={"version":"legacy-computational-v1","matched":False,
            "verification_mode":"reuse_full_request" if variant.iterative_verification else "disabled",
            "definition":"Inference-time flag change; use prepare-ablations/run-ablations for independent matched verification"}
        result.save(dest/f"{name}.json")
    variant=replace(config,multi_column=False)
    detector_kind=payload.get("detector_provenance",{}).get("kind","adapted")
    baseline_path=dest/("zeroed.json" if detector_kind=="source" else "adapted_zeroed.json")
    baseline=Prediction.load(baseline_path)
    expected_detector=payload.get("detector_provenance",{}).get("prediction_hash")
    if expected_detector is not None and expected_detector!=digest({
            "probabilities":baseline.probabilities.tolist(),"metadata":baseline.metadata}):
        raise ValueError("Detector predictions changed; rerun prepare")
    inputs[str(baseline_path)]=sha256(baseline_path)
    no_multi=build_candidates(table,pass2,baseline,variant)
    no_multi["detector_provenance"]=payload.get("detector_provenance",{"kind":detector_kind})
    no_multi["request_hash"]=digest({k:v for k,v in no_multi.items() if k!="request_hash"})
    # Different request means different counterfactual evidence: never misapply a full-model verifier reply.
    result=infer_from_candidates(table,no_multi,None,variant)
    result.metadata["ablation_protocol"]={"version":"legacy-computational-v1","matched":False,
        "verification_mode":"none","definition":"Multi-column rules/FDs and external verification removed together"}
    result.save(dest/"without_multi_column.json")
    for path,expected in inputs.items():
        if sha256(path)!=expected: raise ValueError(f"Inference input changed while running: {path}")
    write_json(out/"frozen"/f"{table.name}.json",{
        "dirty_sha256":table.fingerprint,"source_signature":source_signature(),
        "pass1_hash":digest(load_artifact(table,1)),"pass2_hash":digest(pass2),
        "config":asdict(config),"prediction_files":{p.name:sha256(p) for p in sorted(dest.glob("*.json"))},
        "inference_inputs":inputs,
        "ablation_protocol":{"version":"legacy-computational-v1","matched":False},
        "frozen_before_evaluation":True,"time_unix":time.time()})
    print(f"[{table.name}] froze predictions + 3 legacy computational variants; {len(prediction.repairs)} proposed changes",flush=True)


def evaluate_all(names,out,baseline_status):
    # Imported only after inference has finished/frozen. No reference escapes to model code.
    from zeroclean.evaluation import assess_targets, evaluate
    metrics={};model_info={}
    for name in names:
        table=load_dirty(name)
        frozen_path=out/"frozen"/f"{name}.json"
        if not frozen_path.exists(): raise ValueError(f"Freeze {name} inference before evaluating")
        frozen=json.loads(frozen_path.read_text())
        if frozen.get("dirty_sha256")!=table.fingerprint:
            raise ValueError(f"Dirty input changed after freeze: {name}")
        for path,expected in frozen.get("inference_inputs",{}).items():
            actual=sha256(path) if Path(path).exists() else None
            if actual!=expected: raise ValueError(f"Inference input changed after freeze: {path}")
        metrics[name]={};model_info[name]={}
        for filename,expected_hash in frozen["prediction_files"].items():
            p=out/"predictions"/name/filename
            if sha256(p)!=expected_hash: raise ValueError(f"Prediction changed after freeze: {p}")
            pred=Prediction.load(p)
            threshold=float(pred.metadata.get("config",{}).get("threshold",.5))
            result,failures=evaluate(table,pred,threshold=threshold)
            variant=p.stem
            metrics[name][variant]=result
            model_info[name][variant]=model_metadata(pred.metadata)
            write_json(out/"evaluation_only_failures"/name/f"{variant}.json",failures)
    macro={variant:{metric:sum(metrics[name][variant][metric]["f1"] for name in names)/len(names)
                    for metric in ("detection","repair")}
           for variant in next(iter(metrics.values())) if all(variant in metrics[n] for n in names)}
    summary={**comparison_summary(metrics,assess_targets),
             "targets_met":False,"metric_units":"F1 in [0,1]; gains interpreted as absolute percentage points",
             "aggregation":f"unweighted macro average over {len(names)} measured datasets; target checks require all six",
             "measured_datasets":list(names),"six_dataset_benchmark_complete":set(names)==set(DATASETS),
             "strict_zero_shot":{"benchmark_labels_used_for_inference_or_selection":False,
                                 "labels_used_only_in_evaluation":True,
                                 "failure_reports_must_not_feed_back_into_optimization":True,
                                 "llm_prior_pretraining_contamination":"not measurable"},
             "metrics":metrics,"macro_f1":macro,"model_info":model_info,
             "baseline_status":baseline_status}
    summary["targets_met"]=summary["official_target_assessment"]["met"]
    write_json(out/"summary_metrics.json",summary)
    print(json.dumps({"targets_met":summary["targets_met"],"macro_f1":macro,
                      "official_target_status":summary["official_target_assessment"]["status"]},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","prepare","run","evaluate","all",
                                          "prepare-ablations","run-ablations"),default="all")
    parser.add_argument("--datasets",nargs="+",choices=DATASETS,default=list(DATASETS))
    parser.add_argument("--output",type=Path,default=Path("results"))
    parser.add_argument("--optimize",action="store_true",help="Use dirty-only consistency hill climbing")
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--detector",choices=("adapted","source"),default="adapted",
                        help="Use original-source ZeroED classifier probabilities when ready")
    parser.add_argument("--proposal-file",default="proposals.json",help="Dirty-only proposal artifact name")
    parser.add_argument("--ablation-run",type=Path,
                        help="Exact content-addressed prepared directory for --stage run-ablations")
    args=parser.parse_args()
    out=args.output;out.mkdir(exist_ok=True,parents=True)
    config=LoopConfig(proposal_filename=args.proposal_file)
    if args.stage in ("prepare-ablations","run-ablations"):
        if args.optimize: parser.error("Matched ablations use the fixed configuration recorded during preparation")
        if args.force: parser.error("Matched ablation artifacts are immutable; --force is not supported")
        if args.stage=="run-ablations" and (args.ablation_run is None or len(args.datasets)!=1):
            parser.error("run-ablations requires --ablation-run and exactly one --datasets value")
        for name in args.datasets:
            table=load_dirty(name)
            if args.stage=="prepare-ablations": prepare_ablations(table,out,config,args.detector)
            else:
                from zeroclean.ablations import run_matched_ablations
                run_matched_ablations(table,args.ablation_run,source_signature=matched_source_signature())
                print(f"[{name}] froze matched ablations in {args.ablation_run}",flush=True)
        return
    status=preflight(out)
    if args.stage=="preflight":
        print(json.dumps(status,indent=2));return
    for name in args.datasets:
        table=load_dirty(name)
        if args.stage in ("prepare","all"): prepare(table,out,config,args.force,args.detector)
        if args.stage in ("run","all"): run(table,out,config,args.optimize)
    if args.stage in ("evaluate","all"): evaluate_all(args.datasets,out,status)


if __name__=="__main__": main()
