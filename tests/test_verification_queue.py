import copy
import json
import pytest

from zeroclean.data import DirtyTable
from zeroclean.llm import ResponsePending
from zeroclean.types import digest
from zeroclean.verification import queue_verification

def fixture():
    table=DirtyTable("toy",("a","b"),(("bad","ok"),("fine","fine")),"dirty")
    cells=[{"row":i,"column_index":j,"column":table.columns[j],"original":table.rows[i][j],
            "tuple":dict(zip(table.columns,table.rows[i])),"candidates":[{"value":"better","score":.8}]}
           for i,j in ((0,0),(0,1),(1,0))]
    payload={"dataset":"toy","dirty_sha256":"dirty","model":"gpt-6.1-sol","reasoning_effort":"xhigh",
             "rules":[],"config":{"counterfactual":False},"cells":cells}
    payload["request_hash"]=digest(payload)
    return table,payload

def response(cells):
    return json.dumps({"cells":[{"row":c["row"],"column":c["column"],"original_valid_score":.1,
                                 "selected_value":"better","confidence":.9,"explanation":"dirty context",
                                 "candidates":[{"value":"better","score":.9}]} for c in cells]})

def test_groups_keep_tuple_together_and_preserve_redaction(tmp_path):
    table,payload=fixture();calls=[]
    def request(stage,prompt,metadata):
        evidence=json.loads(prompt.split("\n",1)[1]);calls.append(evidence)
        assert "delta_features" not in prompt and "delta_dimensions" not in evidence
        return response(evidence["cells"])
    result=queue_verification(table,payload,request,tmp_path/"verification.json",batch_size=1)
    assert [len(c["cells"]) for c in calls]==[2,1]
    assert len(result["cells"])==3
    assert result["request_hash"]==payload["request_hash"]

def test_all_groups_enqueued_and_no_partial_output(tmp_path):
    table,payload=fixture();calls=[]
    def pending(stage,prompt,metadata):
        calls.append(metadata);raise ResponsePending("missing")
    dest=tmp_path/"verification.json"
    with pytest.raises(ResponsePending): queue_verification(table,payload,pending,dest,1)
    assert len(calls)==2 and not dest.exists()
    assert json.loads(dest.with_suffix(".status.json").read_text())["pending_groups"]==2

@pytest.mark.parametrize("kind",["missing_cell","missing_candidate","null_score","unknown_candidate"])
def test_incomplete_or_invalid_verification_is_not_committed(tmp_path,kind):
    table,payload=fixture()
    def invalid(stage,prompt,metadata):
        cells=json.loads(prompt.split("\n",1)[1])["cells"]
        out=json.loads(response(cells))
        if kind=="missing_cell": out["cells"].pop()
        if kind=="missing_candidate": out["cells"][0]["candidates"]=[]
        if kind=="null_score": out["cells"][0]["confidence"]=None
        if kind=="unknown_candidate": out["cells"][0]["selected_value"]="invented"
        return json.dumps(out)
    dest=tmp_path/"verification.json"
    with pytest.raises(ValueError,match="exhausted"): queue_verification(table,payload,invalid,dest)
    assert not dest.exists()

def test_corrupted_request_rejected_before_model_call(tmp_path):
    table,payload=fixture();payload["cells"][0]["original"]="changed"
    with pytest.raises(ValueError,match="Corrupted"):
        queue_verification(table,payload,lambda *args:pytest.fail("model called"),tmp_path/"v.json")
