#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zeroclean.data import DATASETS,load_dirty
from zeroclean.llm import FileLLMQueue,ResponsePending
from zeroclean.proposals import queue_cts_proposals
from zeroclean.types import Prediction,digest,write_json

p=argparse.ArgumentParser(description="Queue original-CTS top-3 repair requests for source-detector candidates")
p.add_argument("--datasets",nargs="+",choices=DATASETS,default=list(DATASETS))
p.add_argument("--mask-policy",choices=("all_candidates","query_target"),default="all_candidates")
p.add_argument("--proposal-output-filename",default="proposals_cts.json")
p.add_argument("--payload",type=Path,help="Exact immutable candidate seed; requires one dataset")
p.add_argument("--detection",type=Path,help="Exact source or loop prediction; requires one dataset")
a=p.parse_args();q=FileLLMQueue()
if (a.payload or a.detection) and len(a.datasets)!=1:p.error("Explicit input paths require one dataset")
for name in a.datasets:
    table=load_dirty(name)
    payload=json.loads((a.payload or Path("artifacts/candidates")/f"{name}.json").read_text())
    detection=Prediction.load(a.detection or Path("results/predictions")/name/"zeroed.json")
    requested=[]
    def indexed_request(stage,prompt,metadata):
        key=digest({"stage":stage,"prompt":prompt,"metadata":metadata,
                    "model":q.model,"reasoning_effort":q.effort})
        requested.append({"request_hash":key,"row":metadata["row"],"column":metadata["column"],
                          "attempt":metadata.get("attempt",0)})
        return q.request(stage,prompt,metadata)
    try:
        result=queue_cts_proposals(table,payload,detection,indexed_request,mask_policy=a.mask_policy,
                                  output_filename=a.proposal_output_filename)
        print(name,"completed",len(result["cells"]),flush=True)
    except ResponsePending as e: print(name,str(e),flush=True)
    finally:
        write_json(Path("artifacts/cts_loop")/name/"request_index.json",{
            "dataset":name,"seed_request_hash":payload["request_hash"],"mask_policy":a.mask_policy,
            "candidate_count":len(payload["cells"]),"requests":requested,
            "queue_root":str(q.root)})
