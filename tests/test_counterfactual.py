import numpy as np
import pytest
from zeroclean.data import DirtyTable
from zeroclean.statistics import DirtyIndex
from zeroclean.loop import LoopConfig, build_candidates, infer_from_candidates
from zeroclean.types import Prediction


def toy():
    return DirtyTable("toy",("id","state"),(("a","CA"),("a","CA"),("a","CA"),("a","ca"),("b","NY")),"testhash")


def artifact():
    return {"rules":[{"id":"upper","target":"state","kind":"single","source":"def check(row):\n    return not row['state'].isupper()","description":"uppercase"}],
            "dependencies":[{"determinants":["id"],"dependent":"state","min_support":2,"confidence":.9}],
            "value_repairs":{"state":{"ca":"CA"}}}


def test_loo_support_does_not_support_itself():
    idx=DirtyIndex(toy(),artifact()["dependencies"],discover=False)
    assert idx.conditional(3,1,"CA")[0]==1
    assert idx.conditional(3,1,"ca")[0]==0
    assert idx.conditional(4,1,"NY")[0]==0


def test_candidate_band_includes_high_positives(tmp_path):
    b=Prediction(np.array([[.02,.99],[.02,.30],[.02,.02],[.02,.01],[.02,.02]]))
    payload=build_candidates(toy(),artifact(),b,artifact_root=tmp_path)
    keys={(c["row"],c["column_index"]) for c in payload["cells"]}
    assert (0,1) in keys and (1,1) in keys and (3,1) in keys
    c=next(c for c in payload["cells"] if c["row"]==3)
    repair=next(p for p in c["candidates"] if p["value"]=="CA")
    assert len(repair["delta_features"])==4
    assert repair["delta_features"][0]>0 and repair["delta_features"][1]>0
    pred=infer_from_candidates(toy(),payload)
    assert pred.repairs[3,1]=="CA"


def test_verifier_valid_outlier_and_stale_response(tmp_path):
    payload=build_candidates(toy(),artifact(),artifact_root=tmp_path)
    reply={"request_hash":payload["request_hash"],"model":payload["model"],
           "reasoning_effort":payload["reasoning_effort"],"dirty_sha256":toy().fingerprint,
           "cells":[{"row":3,"column":"state","original_valid_score":.99}]}
    pred=infer_from_candidates(toy(),payload,reply)
    assert (3,1) not in pred.repairs and pred.probabilities[3,1]<.5
    with pytest.raises(ValueError,match="exact candidate request"):
        infer_from_candidates(toy(),payload,{"request_hash":"stale"})


def test_source_candidates_survive_valid_value_prior(tmp_path):
    table=DirtyTable("toy",("value",),(("rare",),("maybe",),("ordinary",)),"testhash")
    baseline=Prediction(np.array([[.99],[.3],[.01]]))
    rules={"rules":[],"valid_values":{"value":["rare","maybe","ordinary"]}}
    payload=build_candidates(table,rules,baseline,artifact_root=tmp_path)
    assert {(c["row"],c["column_index"]) for c in payload["cells"]}=={(0,0),(1,0)}
    assert all(c["base_probability"]<=.12 for c in payload["cells"])


def test_explicit_cts_artifact_cannot_silently_fallback(tmp_path):
    with pytest.raises(ValueError,match="missing"):
        build_candidates(toy(),artifact(),config=LoopConfig(proposal_filename="proposals_cts.json"),artifact_root=tmp_path)
