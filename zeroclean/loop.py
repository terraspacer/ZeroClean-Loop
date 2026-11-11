"""Counterfactual inference. There is deliberately no reference data access here."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import numpy as np
from .data import DirtyTable
from .statistics import DirtyIndex
from .sandbox import evaluate_rules, validate_rules
from .types import Prediction, digest, write_json


@dataclass(frozen=True)
class LoopConfig:
    threshold: float = .5
    candidate_low: float = .25
    candidate_high: float = .85
    top_k: int = 3
    rounds: int = 3
    repair_confidence: float = .8
    counterfactual: bool = True
    multi_column: bool = True
    iterative_verification: bool = True
    seed: int = 42
    proposal_filename: str = "proposals.json"


def sigmoid(x): return 1/(1+math.exp(-max(-30,min(30,x))))
def logit(p): return math.log(max(p,1e-6)/max(1-p,1e-6))


def validate_verification(table,payload,verification):
    """Reject malformed replies before any confidence can influence inference."""
    if verification is None: return {}
    if not isinstance(verification,dict): raise ValueError("Verification must be an object")
    for key,expected in (("request_hash",payload["request_hash"]),("model",payload["model"]),
                         ("reasoning_effort",payload["reasoning_effort"]),("dirty_sha256",table.fingerprint)):
        if verification.get(key)!=expected:
            raise ValueError(f"Verification {key} must match the exact candidate request")
    if "dataset" in verification and verification["dataset"]!=table.name:
        raise ValueError("Verification dataset mismatch")
    source={(c["row"],c["column_index"]):c for c in payload["cells"]}
    replies=verification.get("cells",[])
    if not isinstance(replies,list): raise ValueError("Verification cells must be a list")
    verified={}
    def probability(value,field):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError(f"Verification {field} must be finite and in [0,1]")
    for item in replies:
        if not isinstance(item,dict): raise ValueError("Verification cell must be an object")
        i,col=item.get("row"),item.get("column")
        if type(i) is not int: raise ValueError("Verification row must be an integer")
        if isinstance(col,str):
            if col not in table.columns: raise ValueError("Unknown verification column")
            j=table.columns.index(col)
        elif type(col) is int: j=col
        else: raise ValueError("Verification column must be a name or integer")
        key=i,j
        if key not in source: raise ValueError("Verification coordinate was not requested")
        if key in verified: raise ValueError("Duplicate verification coordinate")
        cell=source[key];allowed={c["value"] for c in cell["candidates"]}
        if "original" in item and item["original"]!=cell["original"]: raise ValueError("Stale verification original value")
        for field in ("original_valid_score","confidence"):
            if field in item and item[field] is not None: probability(item[field],field)
        selected=item.get("selected_value")
        if selected is not None and (not isinstance(selected,str) or selected not in allowed|{cell["original"]}):
            raise ValueError("Verification selected value was not proposed")
        if "joint_repair_required" in item and type(item["joint_repair_required"]) is not bool:
            raise ValueError("Verification joint_repair_required must be boolean")
        scored=item.get("candidates",[])
        if not isinstance(scored,list): raise ValueError("Verification candidates must be a list")
        seen=set()
        for c in scored:
            if not isinstance(c,dict) or not isinstance(c.get("value"),str) or c["value"] not in allowed:
                raise ValueError("Verification candidate value was not proposed")
            if c["value"] in seen: raise ValueError("Duplicate verification candidate")
            seen.add(c["value"])
            if "score" not in c: raise ValueError("Missing verification candidate score")
            probability(c["score"],"score")
            if "joint_repair_required" in c and type(c["joint_repair_required"]) is not bool:
                raise ValueError("Verification joint_repair_required must be boolean")
        verified[key]=item
    return verified


def _cell_proposals(table, artifact_root,filename="proposals.json"):
    if Path(filename).name!=filename: raise ValueError("Proposal filename must be local to its dataset artifact directory")
    path = Path(artifact_root)/table.name/filename
    if not path.exists():
        if filename!="proposals.json": raise ValueError(f"Requested proposal artifact is missing: {path}")
        return {},{"method":"development_nmi","artifact_hash":None}
    body=json.loads(path.read_text())
    if body.get("dirty_sha256") != table.fingerprint:
        raise ValueError("Proposal input fingerprint mismatch")
    if body.get("model")!="gpt-6.1-sol" or body.get("reasoning_effort")!="xhigh":
        raise ValueError("Proposal model provenance mismatch")
    cells=body.get("cells",[])
    out={}
    for cell in cells:
        j=cell["column"]
        if isinstance(j,str):
            if j not in table.columns: raise ValueError("Unknown proposal column")
            j=table.columns.index(j)
        i=cell["row"]
        if type(i) is not int or type(j) is not int: raise ValueError("Proposal coordinates must be integers")
        if not (0<=i<table.shape[0] and 0<=j<table.shape[1]): raise ValueError("Proposal coordinate out of bounds")
        if (i,j) in out: raise ValueError("Duplicate proposal coordinate")
        for c in cell.get("candidates",[]):
            score=c.get("score",.5)
            if not isinstance(c.get("value"),str) or type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
                raise ValueError("Invalid proposal candidate or probability")
        score=cell.get("original_valid_score")
        if score is not None and (type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1):
            raise ValueError("Invalid proposal original validity probability")
        out[i,j]=cell
    return out,{"method":body.get("retrieval_method","development_nmi"),"artifact_hash":digest(body),
                "filename":filename,"request_count":len(body.get("request_hashes",[]))}


def build_candidates(table, artifact, baseline=None, config=LoopConfig(), artifact_root="artifacts/llm"):
    records=table.records()
    proposed=[r for r in artifact.get("rules",[]) if config.multi_column or r.get("kind")=="single"]
    rules,report=validate_rules(proposed,records)
    rules=[r for r in rules if r.get("target") in table.columns]
    matrix=np.asarray(evaluate_rules(rules,records),dtype=np.int8).reshape(len(records),len(rules))
    index=DirtyIndex(table,artifact.get("dependencies",[]) if config.multi_column else [],discover=config.multi_column)
    explicit,proposal_provenance=_cell_proposals(table,artifact_root,config.proposal_filename)
    mapping=artifact.get("value_repairs",{})
    valid=artifact.get("valid_values",{})
    base=np.full(table.shape,.05)
    source_eligible=np.zeros(table.shape,dtype=bool)
    # Baseline probabilities remain available as features; enhanced rules can correct them.
    if baseline is not None:
        baseline.validate(table.shape)
        raw=np.asarray(baseline.probabilities)
        source_eligible=((raw>=config.candidate_low)&(raw<=config.candidate_high))|(raw>=config.threshold)
        base=np.clip(raw,.03,.97).copy()
    target_rules={j:[k for k,r in enumerate(rules) if r["target"]==col] for j,col in enumerate(table.columns)}
    for i,row in enumerate(table.rows):
        for j,col in enumerate(table.columns):
            rr=target_rules[j]
            if rr and (matrix[i,rr]==1).any(): base[i,j]=max(base[i,j],.90)
            fd=index.fd_violation(i,j) if config.multi_column else 0
            if fd: base[i,j]=max(base[i,j],.45+.45*fd)
            if row[j] in mapping.get(col,{}): base[i,j]=max(base[i,j],.95)
            if (i,j) in explicit:
                ps=[float(c.get("score",.5)) for c in explicit[i,j].get("candidates",[]) if c.get("value") != row[j]]
                if ps: base[i,j]=max(base[i,j],max(ps))
            if row[j] in valid.get(col,[]) and not fd and not (rr and (matrix[i,rr]==1).any()): base[i,j]=min(base[i,j],.12)
    # Includes every positive, including >.85, plus uncertainty band [0.25,.85].
    selected=source_eligible|((base>=config.candidate_low)&(base<=config.candidate_high))|(base>=config.threshold)
    cells=[]
    for i,j in np.argwhere(selected):
        i,j=int(i),int(j)
        row,col=table.rows[i],table.columns[j]
        candidates=index.contextual_candidates(i,j,config.top_k)
        if row[j] in mapping.get(col,{}):
            value=mapping[col][row[j]]
            if isinstance(value,str) and value != row[j]:
                candidates.insert(0,{"value":value,"score":.96,"origin":"llm_value_mapping",
                                     "explanation":"Contrastive LLM criteria canonicalization from dirty evidence"})
        special=explicit.get((i,j),{})
        for c in reversed(special.get("candidates",[])):
            if isinstance(c.get("value"),str) and c["value"]!=row[j]:
                candidates.insert(0,dict(c,origin=c.get("origin","llm_context")))
        dedup={}
        for c in candidates:
            if c["value"] not in dedup: dedup[c["value"]]=c
        candidates=list(dedup.values())[:config.top_k]
        cell={"row":i,"column":col,"column_index":j,"original":row[j],"base_probability":float(base[i,j]),
              "tuple":dict(zip(table.columns,row)),"candidates":candidates,
              "original_valid_score":special.get("original_valid_score"),
              "original_rule_violations":int((matrix[i]==1).sum())}
        cells.append(cell)
    # Execute generated checks against counterfactual tuples in bounded batches.
    jobs=[]
    for cell in cells:
        i,j=cell["row"],cell["column_index"]
        before=np.asarray(table.rows[i],dtype=object)
        for c in cell["candidates"]:
            after=before.copy();after[j]=c["value"]
            jobs.append((cell,c,after))
    for start in range(0,len(jobs),1000):
        chunk=jobs[start:start+1000]
        altered=[dict(zip(table.columns,after)) for _,_,after in chunk]
        changed=np.asarray(evaluate_rules(rules,altered),dtype=np.int8).reshape(len(chunk),len(rules))
        for q,(cell,c,after) in enumerate(chunk):
            i,j=cell["row"],cell["column_index"]
            rr=target_rules[j]
            before_bad=int((matrix[i,rr]==1).sum()) if rr else 0
            after_bad=int((changed[q,rr]==1).sum()) if rr else 0
            introduced=bool(((matrix[i]==0)&(changed[q]==1)).any())
            rule_delta=(before_bad-after_bad)/max(len(rr),1)
            fd_delta=index.conditional(i,j,c["value"],after)[0]-index.conditional(i,j,cell["original"])[0] if config.multi_column else 0
            vic_delta=math.tanh((index.vicinity(i,j,c["value"],after)-index.vicinity(i,j,cell["original"]))/4)
            domain_delta=math.tanh((index.domain_support(i,j,c["value"])-index.domain_support(i,j,cell["original"]))/4)
            c["delta_features"]=[float(rule_delta),float(fd_delta),float(vic_delta),float(domain_delta)]
            c["introduced_violation"]=introduced
            c["after_rule_violations"]=int((changed[q]==1).sum())
            c["counterfactual_error"]=bool((changed[q]==-1).any())
    payload={"dataset":table.name,"dirty_sha256":table.fingerprint,"model":"gpt-6.1-sol",
             "reasoning_effort":"xhigh","config":asdict(config),"rules":rules,"rule_validation":report,
             "base_probabilities":base.tolist(),"cells":cells,
             "proposal_provenance":proposal_provenance,
             "delta_dimensions":["rule_satisfaction","fd_support","nmi_vicinity","domain_support"],
             "dependency_count":len(index.dependencies),"artifact_hash":digest(artifact)}
    payload["request_hash"]=digest(payload)
    return payload


def infer_from_candidates(table,payload,verification=None,config=LoopConfig()):
    if payload["dirty_sha256"] != table.fingerprint: raise ValueError("Stale candidate payload")
    expected=digest({k:v for k,v in payload.items() if k!="request_hash"})
    if payload.get("request_hash") != expected: raise ValueError("Candidate payload content hash mismatch")
    verified=validate_verification(table,payload,verification)
    prob=np.asarray(payload["base_probabilities"],dtype=float)
    repairs={}; trace=[]
    # Simultaneous rounds use immutable evidence; accepted repairs never become training labels.
    current=[list(row) for row in table.rows]
    changed_before=set()
    rounds=config.rounds if config.iterative_verification else 1
    for turn in range(rounds):
        changes=0
        for cell in payload["cells"]:
            i,j=cell["row"],cell["column_index"]
            v=verified.get((i,j),{}) if config.iterative_verification else {}
            original_valid=v.get("original_valid_score",cell.get("original_valid_score"))
            selected_original=v.get("selected_value")==table.rows[i][j] and float(v.get("confidence",0))>=.9
            if (original_valid is not None and float(original_valid)>=.9) or selected_original:
                prob[i,j]=min(prob[i,j],.1)
                repairs.pop((i,j),None);current[i][j]=table.rows[i][j]
                continue
            if v and original_valid is not None and float(original_valid)<=.1:
                prob[i,j]=max(prob[i,j],1-float(original_valid))
            if "selected_value" in v and v["selected_value"] is None:
                # An explicit verifier abstention is not permission to take its
                # highest-scoring alternative. Detection remains independent.
                repairs.pop((i,j),None);current[i][j]=table.rows[i][j]
                continue
            ranked=[]
            for c in cell["candidates"]:
                value=c["value"]
                score=float(c.get("score",.5))
                # Independent verifier can approve, reject or select an original outlier.
                cv=next((x for x in v.get("candidates",[]) if x.get("value")==value),None)
                if cv: score=float(cv.get("score",score))
                if v.get("selected_value") is not None:
                    score=float(v.get("confidence",.95)) if value==v["selected_value"] else min(score,.15)
                d=c["delta_features"] if config.counterfactual else [0,0,0,0]
                delta_score=2.2*d[0]+1.8*d[1]+.65*d[2]+.35*d[3]
                inferred=sigmoid(logit(score)+delta_score)
                joint=bool(v.get("joint_repair_required") or (cv or {}).get("joint_repair_required"))
                if config.counterfactual and ((c.get("introduced_violation") and not joint) or c.get("counterfactual_error")):
                    inferred=min(inferred,.1)
                ranked.append((inferred,value,c,delta_score,joint,score))
            if not ranked: continue
            confidence,value,best,delta_score,joint,evidence_confidence=max(ranked,key=lambda x:x[0])
            # Feedback uses both ΔF and independent verification, not an evaluation label.
            p=sigmoid(logit(float(cell["base_probability"]))+.6*delta_score)
            if v and original_valid is not None and float(original_valid)<=.1: p=max(p,1-float(original_valid))
            if v and confidence>=.9: p=max(p,confidence)
            verified_dirty=bool(v and original_valid is not None and float(original_valid)<=.1)
            if config.counterfactual and not joint and not verified_dirty and all(c.get("introduced_violation") for c in cell["candidates"]):
                p=min(p,.45)
            prob[i,j]=p
            if confidence<config.repair_confidence or p<config.threshold: continue
            if current[i][j] != value:
                current[i][j]=value; repairs[i,j]=value;changes+=1
            trace.append({"round":turn,"row":i,"column":table.columns[j],"value":value,
                          "confidence":confidence,"delta_features":best.get("delta_features",[0,0,0,0]),
                          "evidence_confidence":evidence_confidence,"joint_repair_required":joint,
                          "verifier_used":bool(v),"explanation":best.get("explanation","")})
        # Recheck joint tuple consistency after simultaneous edits and reject conflicting edits.
        if config.iterative_verification and repairs:
            indices=sorted({i for i,_ in repairs})
            old_records=[dict(zip(table.columns,table.rows[i])) for i in indices]
            new_records=[dict(zip(table.columns,current[i])) for i in indices]
            rules=payload["rules"]
            old=np.asarray(evaluate_rules(rules,old_records),dtype=np.int8).reshape(len(indices),len(rules))
            new=np.asarray(evaluate_rules(rules,new_records),dtype=np.int8).reshape(len(indices),len(rules))
            for q,i in enumerate(indices):
                if ((old[q]==0)&(new[q]==1)).any() or (new[q]==-1).any():
                    for key in [key for key in repairs if key[0]==i]:
                        repairs.pop(key);current[key[0]][key[1]]=table.rows[key[0]][key[1]]
                        validity=verified.get(key,{}).get("original_valid_score") if config.iterative_verification else None
                        prob[key]=max(prob[key],1-float(validity)) if validity is not None and validity<=.1 else min(prob[key],.45)
        state=frozenset(repairs.items())
        if not changes or state==changed_before: break
        changed_before=state
    metadata={"system":"zeroclean_loop","dirty_sha256":table.fingerprint,"config":asdict(config),
              "request_hash":payload["request_hash"],"rules_accepted":len(payload["rules"]),
              "detector_provenance":payload.get("detector_provenance",{"kind":"adapted","note":"Initial request predates explicit detector hash envelope"}),
              "rules_proposed":len(payload["rule_validation"]),"candidate_cells":len(payload["cells"]),
              "verified_cells":len(verified) if config.iterative_verification else 0,
              "verifier_coverage":len(verified)/max(len(payload["cells"]),1) if config.iterative_verification else 0,
              "rounds_executed":turn+1,"trace":trace,"ground_truth_access":False,
              "proposal_provenance":payload.get("proposal_provenance",{}),
              "fidelity":("Experimental ZeroClean-Loop; original-source ZeroEC CTS proposals with MiniLM "
                          "and source-equivalent distance ranking; additional dirty-only NMI proposals."
                          if payload.get("proposal_provenance",{}).get("method")=="zeroec_source_cts" else
                          "Experimental ZeroClean-Loop; development retrieval uses exact-match NMI, not original ZeroEC CTS.")}
    return Prediction(prob,repairs,metadata).validate(table.shape)
