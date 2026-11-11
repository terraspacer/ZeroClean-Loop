import pytest
from zeroclean.proposals import parse_proposals
from zeroclean.upstream_zeroec import _parse_correction,external_repair_value


def test_top_k_explanation_and_distinctness():
    response='{"original_valid_score":0.05,"candidates":[{"value":"new","score":0.9,"explanation":"same identity peers"}]}'
    assert parse_proposals(response,"old")["candidates"][0]["origin"]=="zeroec_cts_llm"
    with pytest.raises(ValueError,match="distinct"):
        parse_proposals(response,"new")


@pytest.mark.parametrize("value",["NaN","Infinity","true","-0.1","1.1"])
def test_top_k_rejects_invalid_confidence(value):
    with pytest.raises(ValueError):
        parse_proposals('{"original_valid_score":'+value+',"candidates":[]}',"old")


def test_source_null_is_not_a_fabricated_change():
    assert external_repair_value("null","","null")==""
    assert external_repair_value("null","NA","null")=="NA"
    assert external_repair_value("null","bad","bad")==""
    assert external_repair_value("good","bad","bad")=="good"


@pytest.mark.parametrize("response",['{"correction":{}}','{"correction":{"x":[]}}','{"correction":{"x":NaN}}'])
def test_source_correction_requires_requested_scalar(response):
    with pytest.raises(ValueError): _parse_correction(response,"x")


def test_column_traversal_keeps_request_and_artifact_order(tmp_path,monkeypatch):
    import json
    import numpy as np
    from zeroclean.data import DirtyTable
    from zeroclean.types import Prediction,digest
    from zeroclean.proposals import queue_cts_proposals
    import zeroclean.upstream_zeroec as source
    table=DirtyTable("toy",("a","b"),(("one","two"),("three","four")),"dirty")
    cells=[{"row":i,"column_index":j,"column":table.columns[j],"original":table.rows[i][j],
            "tuple":dict(zip(table.columns,table.rows[i])),"candidates":[]}
           for i in range(2) for j in range(2)]
    payload={"dirty_sha256":"dirty","cells":cells};payload["request_hash"]=digest(payload)
    calls=[];ids={}
    class FakeCTS:
        def __init__(self,table,mask,path,exclude_query_row,max_cached_columns):
            assert exclude_query_row and max_cached_columns==2
        def retrieve(self,i,j,k):
            calls.append((i,j));return {"indices":[1-i],"context":"untrusted"}
    def request(stage,prompt,metadata):
        ids[metadata["row"],metadata["column"]]=digest({"stage":stage,"prompt":prompt,"metadata":metadata,
                    "model":"gpt-6.1-sol","reasoning_effort":"xhigh"})
        evidence=json.loads(prompt.split("\n",1)[1])
        assert evidence["observed_dirty_peers"][0]["row"]!=metadata["row"]
        return '{"original_valid_score":0.99,"candidates":[]}'
    monkeypatch.setattr(source,"SourceCTS",FakeCTS);monkeypatch.chdir(tmp_path)
    result=queue_cts_proposals(table,payload,Prediction(np.ones((2,2))),request)
    expected=[(c["row"],c["column"]) for c in cells]
    assert calls==[(0,0),(1,0),(0,1),(1,1)]
    assert [(c["row"],c["column"]) for c in result["cells"]]==expected
    assert result["request_hashes"]==[ids[key] for key in expected]
