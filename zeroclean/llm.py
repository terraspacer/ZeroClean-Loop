"""Content-addressed request/replay transport for authorized experiment subagents."""
from __future__ import annotations
import json
from pathlib import Path
from .types import digest, write_json


class ResponsePending(RuntimeError):
    pass


class FileLLMQueue:
    def __init__(self, root="artifacts/queue", model="gpt-6.1-sol", effort="xhigh"):
        # Original-source adapters temporarily change cwd to satisfy upstream
        # relative data paths. Queue identities/destinations must stay fixed.
        self.root, self.model, self.effort = Path(root).resolve(), model, effort

    def request(self, stage, prompt, metadata):
        body = {"stage": stage, "prompt": prompt, "metadata": metadata,
                "model": self.model, "reasoning_effort": self.effort}
        key = digest(body)
        path = self.root / "requests" / f"{key}.json"
        if not path.exists():
            write_json(path, body)
        response_path = self.root / "responses" / f"{key}.json"
        if not response_path.exists():
            raise ResponsePending(f"Awaiting authorized subagent response: {path}")
        response = json.loads(response_path.read_text())
        if response.get("request_hash") != key or response.get("model") != self.model or response.get("reasoning_effort")!=self.effort:
            raise ValueError("Response provenance mismatch")
        if not isinstance(response.get("response"),str): raise ValueError("LLM response must be text")
        return response["response"]


def load_artifact(table, pass_number=2, root="artifacts/llm"):
    path = Path(root) / table.name / f"pass{pass_number}.json"
    obj = json.loads(path.read_text())
    if obj.get("dirty_sha256") != table.fingerprint:
        raise ValueError(f"Dirty input hash mismatch in {path}")
    if obj.get("model") != "gpt-6.1-sol" or obj.get("reasoning_effort") != "xhigh":
        raise ValueError(f"Unexpected experiment model provenance: {path}")
    return obj
