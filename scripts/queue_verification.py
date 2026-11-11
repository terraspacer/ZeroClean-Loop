#!/usr/bin/env python3
"""Queue or finish hash-bound verification for a full or ablated request."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zeroclean.data import load_dirty
from zeroclean.llm import FileLLMQueue,ResponsePending
from zeroclean.verification import queue_verification
from zeroclean.types import digest, write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload",type=Path,required=True)
    parser.add_argument("--verification-output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=20)
    parser.add_argument("--index-output",type=Path,help="Default: <verification-output stem>.requests.json")
    args=parser.parse_args()
    payload=json.loads(args.payload.read_text())
    table=load_dirty(payload["dataset"])
    queue=FileLLMQueue()
    indexed=[]
    def request(stage,prompt,metadata):
        key=digest({"stage":stage,"prompt":prompt,"metadata":metadata,
                    "model":queue.model,"reasoning_effort":queue.effort})
        indexed.append({"request_hash":key,"group":metadata["group"],
                        "attempt":metadata["attempt"],"cells":metadata["cells"]})
        return queue.request(stage,prompt,metadata)
    try:
        result=queue_verification(table,payload,request,args.verification_output,args.batch_size)
        print(f"Validated {len(result['cells'])} verifier judgments",flush=True)
    except ResponsePending as exc: print(str(exc),flush=True)
    finally:
        write_json(args.index_output or args.verification_output.with_suffix(".requests.json"),{
            "dataset":table.name,"candidate_request_hash":payload["request_hash"],
            "queue_root":str(queue.root),"requests":indexed})

if __name__=="__main__": main()
