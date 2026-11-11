"""Bounded dirty-only hill climbing; reference outcomes are never an objective."""
from __future__ import annotations
from dataclasses import asdict, replace
from .loop import LoopConfig, infer_from_candidates


def proxy_objective(table, prediction, payload):
    """Consistency utility, labeled as a proxy rather than benchmark accuracy."""
    candidates={(c["row"],c["column_index"]):c for c in payload["cells"]}
    trace={(t["row"],table.columns.index(t["column"])):t for t in prediction.metadata.get("trace",[])}
    utility=0.0
    for key,value in prediction.repairs.items():
        cell=candidates[key]
        proposal=next(p for p in cell["candidates"] if p["value"]==value)
        decision=trace.get(key,{})
        d=proposal["delta_features"]
        evidence=float(decision.get("evidence_confidence",proposal.get("score",.5)))
        utility += evidence-.8 + .15*d[0]+.12*d[1]+.03*d[2]
        if (proposal.get("introduced_violation") and not decision.get("joint_repair_required")) or proposal.get("counterfactual_error"): utility -= 2
    # Fixed denominator prevents favoring a tiny subset solely through a ratio.
    return utility/max(len(candidates),1)


def hill_climb(table, payload, verification=None, initial=LoopConfig(), max_steps=3):
    best=initial
    prediction=infer_from_candidates(table,payload,verification,best)
    score=proxy_objective(table,prediction,payload)
    history=[{"config":asdict(best),"proxy":score,"selected":True}]
    seen={(best.threshold,best.repair_confidence)}
    for step in range(max_steps):
        improved=False
        neighbours=[replace(best,threshold=max(.35,min(.7,best.threshold+d))) for d in (-.05,.05)]
        neighbours += [replace(best,repair_confidence=max(.65,min(.98,best.repair_confidence+d))) for d in (-.05,.05)]
        for candidate in neighbours:
            key=(round(candidate.threshold,3),round(candidate.repair_confidence,3))
            if key in seen: continue
            seen.add(key)
            result=infer_from_candidates(table,payload,verification,candidate)
            utility=proxy_objective(table,result,payload)
            selected=utility>score+1e-9
            history.append({"config":asdict(candidate),"proxy":utility,"selected":selected})
            if selected: best,prediction,score,improved=candidate,result,utility,True
        if not improved: break
    return best,prediction,{"objective":"dirty-only consistency utility; not an accuracy estimate",
                            "benchmark_labels_used":False,"history":history,"selected_config":asdict(best)}
