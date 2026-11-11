#!/usr/bin/env python3
"""Resume the official model using validated byte ranges and durable part files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--root",type=Path,default=Path("artifacts/models"))
    a=p.parse_args();a.root.mkdir(exist_ok=True,parents=True)
    final=a.root/"cc.en.300.bin.gz"; parts=a.root/"download_parts";parts.mkdir(exist_ok=True)
    state=a.root/"download_state.json"
    if state.exists(): manifest=json.loads(state.read_text())
    else:
        with urllib.request.urlopen(urllib.request.Request(URL,method="HEAD"),timeout=30) as r:
            total=int(r.headers["Content-Length"])
        prefix=final.stat().st_size if final.exists() else 0
        if prefix>total: raise ValueError("Existing download exceeds source size")
        manifest={"url":URL,"total":total,"prefix":prefix,"chunk_size":32*1024*1024}
        # The known interrupted curl file is a contiguous prefix. Never infer this for parts.
        if prefix: final.rename(parts/"prefix")
        state.write_text(json.dumps(manifest,indent=2)+"\n")
    total=manifest["total"]; prefix=manifest["prefix"]; size=manifest["chunk_size"]
    chunks=[(k,start,min(start+size,total)-1) for k,start in enumerate(range(prefix,total,size))]
    def download(job):
        k,start,end=job;path=parts/f"part{k:05d}"
        for attempt in range(4):
            have=path.stat().st_size if path.exists() else 0
            if have==end-start+1: return k
            if have>end-start+1: raise ValueError("Part exceeds source range")
            try:
                request=urllib.request.Request(URL,headers={"Range":f"bytes={start+have}-{end}"})
                with urllib.request.urlopen(request,timeout=60) as r:
                    expected=f"bytes {start+have}-{end}/{total}"
                    if r.status!=206 or r.headers.get("Content-Range")!=expected:
                        raise ValueError(f"Server did not honor range {expected}")
                    with path.open("ab") as out: shutil.copyfileobj(r,out,1024*1024)
                if path.stat().st_size!=end-start+1: raise IOError("Incomplete range")
                return k
            except Exception:
                if attempt==3: raise
                time.sleep(min(attempt+1,3))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for count,future in enumerate(as_completed([pool.submit(download,j) for j in chunks]),1):
            future.result()
            print(f"Completed {count}/{len(chunks)} ranges",flush=True)
    combined=final.with_suffix(".gz.assembling")
    with combined.open("wb") as out:
        if prefix:
            with (parts/"prefix").open("rb") as src: shutil.copyfileobj(src,out,8*1024*1024)
        for k,_,_ in chunks:
            with (parts/f"part{k:05d}").open("rb") as src: shutil.copyfileobj(src,out,8*1024*1024)
    if combined.stat().st_size!=total: raise ValueError("Assembled length mismatch")
    combined.replace(final)
    binary=a.root/"cc.en.300.bin"
    temporary=binary.with_suffix(".bin.extracting")
    # gzip verifies the source CRC at EOF; never install a partially extracted model.
    with gzip.open(final,"rb") as src,temporary.open("wb") as out:
        shutil.copyfileobj(src,out,8*1024*1024)
    temporary.replace(binary)
    print(f"Verified gzip and extracted {binary.stat().st_size} bytes to {binary}",flush=True)


if __name__=="__main__": main()
