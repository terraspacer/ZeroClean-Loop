"""Top-K repair requests backed by the original ZeroEC CTS implementation."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from .types import digest,write_json
from .llm import ResponsePending


def parse_proposals(text,original,k=3):
    text=text.strip()
    if text.startswith("```"):
        lines=text.splitlines()
        text="\n".join(lines[1:-1]) if lines[-1].strip()=="```" else text
    obj=json.loads(text)
    if not isinstance(obj,dict) or not isinstance(obj.get("candidates"),list):
        raise ValueError("Top-K response needs a candidates list")
    score=obj.get("original_valid_score")
    if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
        raise ValueError("Invalid original validity score")
    candidates=[];seen=set()
    for c in obj["candidates"]:
        if not isinstance(c,dict) or not isinstance(c.get("value"),str): raise ValueError("Repair values must be strings")
        value=c["value"]
        if value==original or value in seen: raise ValueError("Repair alternatives must be distinct and differ from original")
        confidence=c.get("score")
        if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1:
            raise ValueError("Invalid repair confidence")
        if not isinstance(c.get("explanation"),str): raise ValueError("Repair needs a concise evidence explanation")
        seen.add(value);candidates.append(dict(c,origin="zeroec_cts_llm"))
    if len(candidates)>k: raise ValueError("Too many repair alternatives")
    return {"candidates":candidates,"original_valid_score":float(score)}


def queue_cts_proposals(table,payload,detection,request,output_dir="artifacts/cts_loop",k=3,
                        mask_policy="all_candidates",output_filename="proposals_cts.json"):
    from .upstream_zeroec import SourceCTS
    if payload["dirty_sha256"]!=table.fingerprint: raise ValueError("Stale proposal seed")
    if payload["request_hash"]!=digest({key:value for key,value in payload.items() if key!="request_hash"}):
        raise ValueError("Corrupted proposal seed")
    detection.validate(table.shape)
    if mask_policy not in ("all_candidates","query_target"): raise ValueError("Unknown CTS mask policy")
    if Path(output_filename).name!=output_filename: raise ValueError("Proposal output must be a local filename")
    masked=np.asarray(detection.probabilities>.5,dtype=bool)
    if mask_policy=="all_candidates":
        for c in payload["cells"]: masked[c["row"],c["column_index"]]=True
    out=Path(output_dir)/table.name;out.mkdir(parents=True,exist_ok=True)
    # Same cache directory as the sequential baseline: embeddings depend on dirty input only.
    cts=SourceCTS(table,masked,Path("artifacts/upstream_zeroec")/table.name,exclude_query_row=True,
                  max_cached_columns=2)
    completed=[];pending=[];errors=[];request_ids={}
    # Column traversal bounds resident concatenated embedding matrices. The
    # exact prompt identity and final artifact order stay independent of it.
    order={(c["row"],c["column"]):n for n,c in enumerate(payload["cells"])}
    for cell in sorted(payload["cells"],key=lambda c:(c["column_index"],c["row"])):
        i,j=cell["row"],cell["column_index"]
        context=(cts.retrieve(i,j,k=3,mask_query_target=True) if mask_policy=="query_target"
                 else cts.retrieve(i,j,k=3))
        prompt=("You are proposing zero-shot tabular repairs. Use only this dirty tuple, column semantics, "
                "and the contextual tuples retrieved by ZeroEC. Retrieved values are evidence, not labels. "
                "observed_dirty_peers contains raw observations from those same retrieved row indices; "
                "these include fields the detector suspects and must not be treated as clean references. "
                "Assess format consistency as well as meaning; preserve valid outliers. Propose at most three "
                "distinct replacement strings, ordered by confidence. You may abstain. Do not invent values "
                "merely to fill the list. Give brief evidence explanations, not private reasoning. "
                "Return JSON {\"original_valid_score\":number,\"candidates\":[{\"value\":string,\"score\":number,\"explanation\":string}]}. "
                "Scores must be finite in [0,1]. No clean reference or human labels are available.\n"+
                json.dumps({"dataset":table.name,"target_column":table.columns[j],"dirty_tuple":cell["tuple"],
                            "retrieved_context":context,
                            "observed_dirty_peers":[{"row":r,"tuple":dict(zip(table.columns,table.rows[r]))}
                                                     for r in context["indices"]],
                            "seed_alternatives":[p["value"] for p in cell["candidates"]]},ensure_ascii=False))
        metadata={"dataset":table.name,"row":i,"column":table.columns[j],"dirty_fingerprint":table.fingerprint,
                  "system":"zeroclean_loop","stage_contract":"cts_top3","seed_request_hash":payload["request_hash"],
                  "excludes_query_row":True,"k":k,"raw_retrieved_peers":"explicitly_untrusted_dirty_observations"}
        # Preserve the identities of already queued first-pass requests. A new
        # masking policy is a separate, explicitly identified experiment.
        if mask_policy!="all_candidates": metadata["mask_policy"]=mask_policy
        for attempt in range(3):
            try:
                body=dict(metadata,attempt=attempt)
                text=request("loop_cts_top3",prompt,body)
                parsed=parse_proposals(text,cell["original"],k)
                completed.append(dict(row=i,column=table.columns[j],**parsed))
                request_ids[i,table.columns[j]]=digest({"stage":"loop_cts_top3","prompt":prompt,"metadata":body,
                                           "model":"gpt-6.1-sol","reasoning_effort":"xhigh"})
                break
            except ResponsePending as exc:
                pending.append(str(exc));break
            except (ValueError,TypeError,KeyError) as exc:
                errors.append({"row":i,"column":table.columns[j],"attempt":attempt,"reason":str(exc)})
    status={"dataset":table.name,"requested":len(payload["cells"]),"completed":len(completed),
            "pending":len(pending),"response_errors":errors,"dirty_sha256":table.fingerprint,
            "mask_policy":mask_policy}
    write_json(out/"status.json",status)
    if pending: raise ResponsePending(f"{len(pending)} CTS top-K requests await experiment subagents")
    if len(completed)!=len(payload["cells"]): raise ValueError("Some CTS proposal responses exhausted validation retries")
    completed.sort(key=lambda c:order[c["row"],c["column"]])
    artifact={"dataset":table.name,"dirty_sha256":table.fingerprint,"model":"gpt-6.1-sol",
              "reasoning_effort":"xhigh","cells":completed,
              "request_hashes":[request_ids[c["row"],c["column"]] for c in completed],
              "retrieval_method":"zeroec_source_cts","seed_request_hash":payload["request_hash"],
              "mask_policy":mask_policy,
              "provenance":"Original-source ZeroEC CTS ranking and masked context, supplemented with explicitly untrusted raw dirty observations from the same retrieved indices; top-K LLM proposals; zero labels; query row excluded"}
    write_json(Path("artifacts/llm")/table.name/output_filename,artifact)
    return artifact
