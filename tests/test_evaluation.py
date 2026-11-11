import numpy as np
import pytest
from zeroclean.data import DirtyTable, load_dirty
from zeroclean.types import Prediction
from zeroclean.evaluation import evaluate, prf, assess_targets


def test_wrong_repairs_and_clean_unchanged_are_not_true_positives(tmp_path):
    table=DirtyTable("toy",("x","y"),(("bad","ok"),("bad2","good")),"hash")
    (tmp_path/"toy_clean.csv").write_text("x,y\nright,ok\nright2,good\n")
    prediction=Prediction(np.array([[.9,.1],[.9,.9]]),{(0,0):"right",(1,0):"stillwrong",(1,1):"spoiled"})
    result,_=evaluate(table,prediction,root=tmp_path)
    assert result["detection"]==prf(2,3,2)
    assert result["repair"]==prf(1,3,2)
    assert result["repair"]["f1"]==.4


def test_repair_without_detection_is_rejected(tmp_path):
    table=DirtyTable("toy",("x",),(("bad",),),"hash")
    (tmp_path/"toy_clean.csv").write_text("x\nright\n")
    with pytest.raises(ValueError,match="repair must"):
        evaluate(table,Prediction(np.array([[.1]]),{(0,0):"right"}),root=tmp_path)


def test_preserve_empty_zero_padding_and_literal_na(tmp_path):
    (tmp_path/"hospital_error-01.csv").write_text("a,b,c\n001,,NA\n")
    t=load_dirty("hospital",tmp_path)
    assert t.rows==(("001","","NA"),)
    assert not hasattr(t,"clean")


def test_no_predictions_and_reference_alignment(tmp_path):
    assert prf(0,0,10)["f1"]==0
    table=DirtyTable("toy",("x",),(("bad",),),"h")
    (tmp_path/"toy_clean.csv").write_text("other\nright\n")
    with pytest.raises(ValueError,match="alignment"):
        evaluate(table,Prediction(np.array([[.1]])),root=tmp_path)


def test_missing_baseline_cannot_pass_targets():
    result=assess_targets({})
    assert result["met"] is False
    assert result["status"]=="unverified_missing_baselines_or_runs"
