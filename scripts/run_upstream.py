#!/usr/bin/env python3
"""Run/resume original-source baselines; pending LLM requests stay inspectable."""
from pathlib import Path
import argparse
import fcntl
import gc
import json
import os
import sys
if __name__=="__main__" and os.environ.get("PYTHONHASHSEED")!="0":
    os.execve(sys.executable,[sys.executable,*sys.argv],dict(os.environ,PYTHONHASHSEED="0"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zeroclean.baselines import (BaselineUnavailable,queue_zeroed_preliminary,
    run_exact_zeroed,run_reconstructed_zeroed_zeroec)
from zeroclean.data import DATASETS,load_dirty
from zeroclean.llm import FileLLMQueue,ResponsePending
from zeroclean.types import Prediction,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preliminary","zeroed","zeroec"),required=True)
    parser.add_argument("--datasets",nargs="+",choices=DATASETS,default=list(DATASETS))
    parser.add_argument("--fasttext",type=Path,default=Path("artifacts/models/cc.en.300.bin"))
    parser.add_argument("--output",type=Path,default=Path("results"))
    parser.add_argument("--prompt-profile",choices=("committed_zeroec","author_zerodc_29cde9e0dbcd"),
                        default="author_zerodc_29cde9e0dbcd")
    args=parser.parse_args()
    queue=FileLLMQueue();status={}
    for name in args.datasets:
        table=load_dirty(name);work=Path("artifacts/upstream_zeroed")/name
        try:
            if args.stage=="preliminary":
                status[name]=queue_zeroed_preliminary(table,queue.request,work)
            elif args.stage=="zeroed":
                model=args.fasttext
                reduced=Path("artifacts/models/reduced")/f"cc.en.{table.shape[1]}.bin"
                if reduced.exists():model=reduced
                elif args.fasttext==Path("artifacts/models/cc.en.300.bin"):
                    raise BaselineUnavailable("Prepare bounded-memory dimensional models with scripts/prepare_fasttext_dimensions.py first")
                result=run_exact_zeroed(table,queue.request,work,fasttext_path=model)
                result.save(args.output/"predictions"/name/"zeroed.json")
                status[name]={"status":"complete","fidelity":result.metadata.get("fidelity")}
            else:
                detection=Prediction.load(args.output/"predictions"/name/"zeroed.json")
                result=run_reconstructed_zeroed_zeroec(table,detection,queue.request,
                    Path("artifacts/upstream_zeroec")/name,use_general_examples=False,prompt_profile=args.prompt_profile)
                result.save(args.output/"predictions"/name/"zeroed_zeroec_reconstruction.json")
                status[name]={"status":"complete","fidelity":result.metadata.get("fidelity")}
        except ResponsePending as exc:
            status[name]={"status":"awaiting_subagent_responses","reason":str(exc)}
        except (BaselineUnavailable,FileNotFoundError) as exc:
            status[name]={"status":"missing_resource","reason":str(exc)}
        print(name,json.dumps(status[name]),flush=True)
        gc.collect()
    # Dataset workers run independently; retain every worker's stage status.
    args.output.mkdir(parents=True,exist_ok=True)
    status_path=args.output/f"upstream_{args.stage}_status.json"
    with (args.output/f".upstream_{args.stage}_status.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        merged=json.loads(status_path.read_text()) if status_path.exists() else {}
        merged.update(status)
        write_json(status_path,merged)


if __name__=="__main__":main()
