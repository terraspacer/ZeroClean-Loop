"""Batched, content-addressed verifier requests using dirty evidence only."""
from __future__ import annotations

import json
from pathlib import Path

from .llm import ResponsePending
from .loop import validate_verification
from .types import digest, write_json


def _parse(text):
    text=text.strip()
    if text.startswith("```") and text.endswith("```"):
        text="\n".join(text.splitlines()[1:-1])
    value=json.loads(text)
    if not isinstance(value,dict) or not isinstance(value.get("cells"),list):
        raise ValueError("Verifier must return an object with a cells list")
    return value["cells"]


def queue_verification(table,payload,request,output_path,batch_size=20):
    """Finish only after every exact candidate has a validated response.

    The request contains the variant's own evidence. In particular a redacted
    ablation is never supplemented with a hidden full-model feature payload.
    """
    if type(batch_size) is not int or batch_size<1:
        raise ValueError("batch_size must be a positive integer")
    if payload.get("dirty_sha256")!=table.fingerprint:
        raise ValueError("Verification input fingerprint mismatch")
    if payload.get("request_hash")!=digest({k:v for k,v in payload.items() if k!="request_hash"}):
        raise ValueError("Corrupted verification request")
    envelope={key:payload[key] for key in ("request_hash","model","reasoning_effort","dirty_sha256","dataset")}
    combined=[];pending=[];errors=[];requests=[]
    # Do not split cells from one tuple across groups: joint changes are visible
    # together. A single wide tuple may exceed the requested batch size.
    by_row={}
    for cell in payload["cells"]:
        by_row.setdefault(cell["row"],[]).append(cell)
    groups=[];group=[]
    for row in sorted(by_row):
        cells=by_row[row]
        if group and len(group)+len(cells)>batch_size:
            groups.append(group);group=[]
        group.extend(cells)
    if group: groups.append(group)
    for number,cells in enumerate(groups):
        evidence={key:payload[key] for key in ("dataset","dirty_sha256","rules","delta_dimensions","config") if key in payload}
        evidence["cells"]=cells
        prompt=("Verify these proposed tabular repairs using only the supplied dirty tuples, executable criteria, "
                "candidate evidence and any counterfactual features present in this request. "
                "There are no clean values or human labels. Repeated peer values and rules are fallible evidence. "
                "If tentative_tuple or counterfactual_before_tuple is present, it contains prior model-proposed "
                "changes and is untrusted, not a new observation or a clean label. The original dirty tuple "
                "remains in tuple. Reassess earlier proposals for mutual consistency; do not assume they are correct. "
                "Check both semantic validity and the table's supported representation; preserve valid outliers. "
                "A value can be valid in isolation yet inconsistent with its tuple or the column's supported "
                "representation. Proposal scores, original_valid_score and base_probability are fallible model "
                "estimates, not labels: assess them independently against the supplied evidence. "
                "Review every cell and every candidate. A repair may require other proposed changes in the same tuple; "
                "set joint_repair_required only when the complete proposed tuple is coherent. "
                "You may reject all alternatives while judging the original invalid. "
                "Return JSON {\"cells\":[{\"row\":integer,\"column\":string,\"original_valid_score\":number,"
                "\"selected_value\":string_or_null,\"confidence\":number,\"explanation\":string,"
                "\"joint_repair_required\":boolean,\"candidates\":[{\"value\":string,\"score\":number}]}]}. "
                "Use a supplied alternative or the original value as selected_value; null explicitly abstains from repair. "
                "Scores are finite probabilities in [0,1]. Include every requested cell exactly once and score every "
                "alternative exactly once. Give brief evidence explanations, not private reasoning.\n"+
                json.dumps(evidence,ensure_ascii=False,allow_nan=False))
        subset=dict(payload,cells=cells)
        for attempt in range(3):
            metadata={"dataset":table.name,"dirty_fingerprint":table.fingerprint,
                      "system":"zeroclean_loop","stage_contract":"counterfactual_verification",
                      "candidate_request_hash":payload["request_hash"],"group":number,
                      "cells":[[c["row"],c["column_index"]] for c in cells],"attempt":attempt}
            try:
                reply=_parse(request("loop_verification",prompt,metadata))
                verified=validate_verification(table,subset,dict(envelope,cells=reply))
                if len(verified)!=len(cells): raise ValueError("Incomplete verifier cell coverage")
                for c in cells:
                    item=verified[c["row"],c["column_index"]]
                    for field in ("original_valid_score","selected_value","confidence","explanation"):
                        if field not in item: raise ValueError(f"Missing verifier {field}")
                    if item["original_valid_score"] is None or item["confidence"] is None:
                        raise ValueError("Verifier probabilities must not be null")
                    if not isinstance(item["explanation"],str): raise ValueError("Verifier explanation must be text")
                    if {v["value"] for v in item.get("candidates",[])}!={v["value"] for v in c["candidates"]}:
                        raise ValueError("Incomplete verifier candidate coverage")
                combined.extend(reply)
                requests.append(digest({"stage":"loop_verification","prompt":prompt,"metadata":metadata,
                                        "model":payload["model"],"reasoning_effort":payload["reasoning_effort"]}))
                break
            except ResponsePending as exc:
                pending.append(str(exc));break
            except (ValueError,TypeError,KeyError) as exc:
                errors.append({"group":number,"attempt":attempt,"reason":str(exc)})
    output_path=Path(output_path)
    status={"candidate_request_hash":payload["request_hash"],"groups":len(groups),"pending_groups":len(pending),
            "requested_cells":len(payload["cells"]),"completed_cells":len(combined),"response_errors":errors}
    write_json(output_path.with_suffix(".status.json"),status)
    if pending: raise ResponsePending(f"{len(pending)} verifier groups await experiment subagents")
    if len(combined)!=len(payload["cells"]): raise ValueError("Verifier retries exhausted before full coverage")
    result=dict(envelope,cells=combined,queue_request_hashes=requests,
                provenance="Independent batched verification of this exact variant; all tuple cells grouped together; no reference labels")
    validate_verification(table,payload,result)
    if output_path.exists():
        if json.loads(output_path.read_text())!=result:
            raise ValueError("Refusing to overwrite a different verifier artifact")
    else: write_json(output_path,result)
    return result
