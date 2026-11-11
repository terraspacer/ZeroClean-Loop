import json
import pytest
from zeroclean.llm import FileLLMQueue,ResponsePending
from zeroclean.types import digest


def test_relative_queue_survives_upstream_working_directory_change(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    queue=FileLLMQueue("queue")
    with pytest.raises(ResponsePending): queue.request("stage","prompt",{"dirty":"hash"})
    path=next((tmp_path/"queue/requests").glob("*.json"))
    body=json.loads(path.read_text())
    response_dir=tmp_path/"queue/responses";response_dir.mkdir()
    (response_dir/path.name).write_text(json.dumps({"request_hash":digest(body),"model":queue.model,
        "reasoning_effort":queue.effort,"response":"answer"}))
    work=tmp_path/"source_work";work.mkdir();monkeypatch.chdir(work)
    assert queue.request("stage","prompt",{"dirty":"hash"})=="answer"
    assert not (work/"queue").exists()


def test_concurrent_identical_requests_have_one_valid_artifact(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    queue=FileLLMQueue(tmp_path/"queue")
    def call(_):
        with pytest.raises(ResponsePending): queue.request("stage","same prompt",{"dirty":"hash"})
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(call,range(50)))
    files=list((tmp_path/"queue/requests").glob("*.json"))
    assert len(files)==1
    assert files[0].stem==digest(json.loads(files[0].read_text()))
    assert not list((tmp_path/"queue/requests").glob("*.tmp"))
