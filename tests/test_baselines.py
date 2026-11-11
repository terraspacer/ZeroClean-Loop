"""Fidelity boundaries and inference-only baseline contracts; synthetic inputs only."""
from pathlib import Path
import json

import numpy as np
import pytest

from zeroclean.baselines import (BaselineUnavailable, adapted_zeroed,
                                 adapted_zeroed_zeroec, preflight_zeroec, queue_zeroed_preliminary,
                                 run_exact_zeroed,
                                 run_reconstructed_zeroed_zeroec)
from zeroclean.data import DirtyTable
from zeroclean.types import Prediction
from zeroclean.upstream_zeroed import _definitions, _ProbabilityCapture
from zeroclean.upstream_zeroec import SourceCTS, nearest_rows, normalized_mutual_information


def test_exact_runner_fails_closed_before_any_llm_or_dataset_write(tmp_path):
    calls = []
    table = DirtyTable("toy", ("x",), (("dirty",),), "synthetic")
    with pytest.raises(BaselineUnavailable):
        run_exact_zeroed(table, lambda *args: calls.append(args), tmp_path / "output",
                         source_dir=tmp_path / "absent-source")
    assert calls == []
    assert (tmp_path / "output" / "preflight.json").is_file()
    assert not (tmp_path / "output" / "source_work").exists()


def test_extraction_does_not_execute_entrypoint_or_compile_evaluation(tmp_path):
    source = tmp_path / "upstream.py"
    source.write_text('''
import absent_runtime_dependency
raise RuntimeError("top-level source execution is forbidden")
def infer(value):
    return value

def read_reference():
    raise RuntimeError("forbidden")

if __name__ == "__main__":
    raise RuntimeError("entrypoint execution is forbidden")
''')
    namespace = _definitions(source, {"__name__": "extracted_test"}, {"read_reference"})
    assert namespace["infer"]("dirty") == "dirty"
    assert "read_reference" not in namespace


def test_zeroed_preliminary_warmup_requests_match_full_source_runner(tmp_path):
    pytest.importorskip("fasttext")
    pytest.importorskip("tqdm")
    from zeroclean.llm import FileLLMQueue, ResponsePending
    table = DirtyTable("toy", ("x", "y"), (("a", "1"), ("a", "1"), ("b", "2")), "synthetic")
    queue = FileLLMQueue(tmp_path / "queue")
    report = queue_zeroed_preliminary(table, queue.request, tmp_path / "run")
    assert set(report["pending_columns"]) == {"x", "y"}
    assert report["generated_code_executed"] is False
    before = {path.name for path in (tmp_path / "queue" / "requests").glob("*.json")}
    assert len(before) == 2
    # The first stage yields before any model use. This sentinel is deliberately
    # not a usable model; a model-loading call would make the test fail.
    sentinel = tmp_path / "not-a-model.bin"
    sentinel.write_bytes(b"not an embedding model")
    with pytest.raises(ResponsePending, match="all available requests were enqueued"):
        run_exact_zeroed(table, queue.request, tmp_path / "run", fasttext_path=sentinel)
    after = {path.name for path in (tmp_path / "queue" / "requests").glob("*.json")}
    assert after == before


def test_source_probability_observer_preserves_prediction_and_exact_half_tie():
    class Classifier:
        classes_ = np.array([0, 1])
        calls = []
        def predict(self, features):
            self.calls.append(("predict", features))
            return np.array([0, 0, 1])
        def predict_proba(self, features):
            self.calls.append(("predict_proba", features))
            return np.array([[.75, .25], [.5, .5], [.15, .85]])
    model = Classifier()
    features = np.arange(6).reshape(3, 2)
    observer = _ProbabilityCapture(model)
    assert observer.predict(features).tolist() == [0, 0, 1]
    assert observer.probabilities.tolist() == [.25, .5, .85]
    assert observer.decisions.tolist() == [0, 0, 1]
    assert observer.score_kind == "mlp_class_1_predict_proba"
    assert [name for name, _ in model.calls] == ["predict", "predict_proba"]
    assert all(value is features for _, value in model.calls)


def test_source_all_clean_representatives_can_leave_only_synthetic_error_training(tmp_path):
    import ast
    from collections import defaultdict
    import io
    import os
    import re
    from types import SimpleNamespace
    import pandas as pd
    source = Path("vendor/ZeroED/main.py")
    frame = pd.DataFrame({"x": ["a", "b"]})
    namespace = {"__name__": "synthetic_source_label_trace", "os": os, "ast": ast, "re": re,
                 "np": np, "pd": pd, "defaultdict": defaultdict, "funcs_directory": str(tmp_path),
                 "dirty_csv": frame, "related_attrs_dict": {"x": {}}, "center_index_value_label_dict": {},
                 "para_file": io.StringIO(), "tqdm": lambda values, **kwargs: values}
    native = _definitions(source, namespace)
    centers = {"x": [{"x": "a"}, {"x": "b"}]}
    clusters = {"x": [[0, 1], [0], [1]]}
    (tmp_path / "error_checking_x.txt").write_text(json.dumps({"column_name": "x", "entries": [
        {"value_row": str(value), "error_analysis": "synthetic valid value", "has_error_in_x_value": False}
        for value in centers["x"]]}, indent=2))
    labels = native["extract_llm_label_res"](["x"], str(tmp_path), clusters, centers)
    assert [item[2] for item in labels["x"]] == [0, 0]
    # This is the committed source behavior: no labeled errors means no
    # error-conditioned clean predicates, rather than reuse of preliminaries.
    functions = native["task_func_gen"]("x", {})
    assert functions == {"x": {"clean": [], "dirty": []}}
    native.update({"all_attrs": ["x"], "funcs_for_attr": functions,
                   "det_right_list": [(0, "x"), (1, "x")],
                   "init_det_right_dict": {"x": list(enumerate(centers["x"]))}})
    loops = [node for node in ast.walk(ast.parse(source.read_text()))
             if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "attr"
             and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                     and isinstance(call.func.value, ast.Name) and call.func.value.id == "det_right_list"
                     and call.func.attr == "remove" for call in ast.walk(node))]
    assert len(loops) == 1
    exec(compile(ast.Module(body=loops, type_ignores=[]), "<source-zero-predicate-filter>", "exec"), native)
    assert native["det_right_list"] == []
    _, training_labels = native["prep_train_feat"]("x", frame, [], [], {"x": {}},
        {"x": {"dirty": [{"x": "synthetic-invalid"}]}}, functions,
        SimpleNamespace(get_word_vector=lambda value: np.array([.5], dtype=np.float32)), None, str(tmp_path))
    assert training_labels == [1]


@pytest.mark.parametrize("label", [0, 1])
def test_source_probability_observer_uses_constant_for_one_class(label):
    class Classifier:
        classes_ = np.array([label])
        def predict(self, features):
            return np.full(len(features), label)
        def predict_proba(self, features):
            raise AssertionError("One-class sklearn sigmoid output must not be interpreted as a class-1 probability")
    observer = _ProbabilityCapture(Classifier())
    assert observer.predict(np.zeros((3, 2))).tolist() == [label] * 3
    assert observer.probabilities.tolist() == [float(label)] * 3
    assert observer.score_kind == "constant_pseudo_label_class"


def test_prepared_fasttext_matches_upstream_pca_and_immutable_cache(tmp_path, monkeypatch):
    fasttext = pytest.importorskip("fasttext")
    import fasttext.util
    from zeroclean.fasttext_resources import prepare_fasttext_dimensions, prepared_fasttext_provenance
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta gamma delta epsilon zeta eta theta\n" * 60)
    original = tmp_path / "original.bin"
    model = fasttext.train_unsupervised(str(corpus), model="skipgram", dim=24,
                                        minCount=1, epoch=2, bucket=9000, thread=1, verbose=0)
    model.save_model(str(original))
    del model
    before = np.random.get_state()
    # Preparation must never load even one full original C++ model. Only the
    # comparison below is allowed to use load_model on this tiny fixture.
    original_load = fasttext.load_model
    with monkeypatch.context() as patch:
        patch.setattr(fasttext, "load_model", lambda *args: (_ for _ in ()).throw(AssertionError("full load")))
        manifest = prepare_fasttext_dimensions(original, tmp_path / "reduced")
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    assert manifest["rng_consumed"] is False
    assert manifest["identity"]["projection_block_rows"] == 8192
    assert manifest["identity"]["algorithm"] == "upstream_pca_complex_row_blocks_real_full_matmul_v2"
    assert set(manifest["dimensions"]) == {"6", "9", "11", "17", "22"}
    for dimension, entry in manifest["dimensions"].items():
        expected = original_load(str(original))
        fasttext.util.reduce_model(expected, int(dimension))
        got = original_load(entry["path"])
        assert got.get_dimension() == int(dimension)
        assert np.array_equal(got.get_input_matrix(), expected.get_input_matrix())
        assert np.array_equal(got.get_output_matrix(), expected.get_output_matrix())
        assert np.array_equal(got.get_word_vector("alpha"), expected.get_word_vector("alpha"))
        expected_path = tmp_path / "expected.bin"
        expected.save_model(str(expected_path))
        assert Path(entry["path"]).read_bytes() == expected_path.read_bytes()
        assert not (Path(entry["path"]).stat().st_mode & 0o222)
        provenance = prepared_fasttext_provenance(entry["path"])
        assert provenance["model_sha256"] == entry["sha256"]
        assert provenance["original_sha256"] == manifest["identity"]["original_sha256"]
        del got, expected
    repeated = prepare_fasttext_dimensions(original, tmp_path / "reduced")
    assert repeated == manifest
    damaged = Path(manifest["dimensions"]["6"]["path"])
    damaged.chmod(0o644)
    with damaged.open("r+b") as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="Immutable fastText cache"):
        prepare_fasttext_dimensions(original, tmp_path / "reduced")


@pytest.mark.parametrize("dimension", [6, 9, 11, 17, 22])
@pytest.mark.parametrize("complex_projection", [False, True])
@pytest.mark.filterwarnings("ignore:Casting complex values to real discards the imaginary part")
def test_bounded_fasttext_projection_bytes_match_upstream_over_multiple_blocks(
        tmp_path, monkeypatch, dimension, complex_projection):
    pytest.importorskip("fasttext")
    from fasttext.util.util import _reduce_matrix
    from zeroclean.fasttext_resources import _write_projected_matrix
    rng = np.random.RandomState(712)
    # The original width300 and two full blocks plus an odd tail exercise both
    # dtype promotion and shape-dependent BLAS kernels absent in tiny fixtures.
    matrix = rng.normal(size=(16421, 300)).astype(np.float32)
    basis = rng.normal(size=(300, 300)).astype(np.float32)
    if complex_projection:
        basis = (basis + 1j * rng.normal(size=basis.shape).astype(np.float32)).astype(np.complex64)
    # np.linalg.eig returns the full basis; upstream slices its first d columns
    # without making them contiguous. Preserve that leading dimension as well.
    projection = basis[:, :dimension]
    assert projection.strides[0] == 300 * projection.dtype.itemsize
    expected, _ = _reduce_matrix(matrix, dimension, projection)
    expected = np.asarray(expected, dtype=np.float32)
    calls = []
    matmul = np.matmul
    def observe(left, right):
        calls.append(left.shape[0])
        return matmul(left, right)
    path = tmp_path / "projection.bin"
    with monkeypatch.context() as patch:
        patch.setattr(np, "matmul", observe)
        with path.open("wb") as handle:
            _write_projected_matrix(matrix, projection, handle)
    assert path.read_bytes() == expected.tobytes()
    assert calls == ([8192, 8192, 37] if complex_projection else [16421])


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
def test_adaptation_sandbox_mlp_is_deterministic_and_has_no_repairs():
    rows = tuple((("b" if i % 5 == 0 else "g") + str(i), str(i % 7)) for i in range(100))
    table = DirtyTable("toy", ("x", "context"), rows, "synthetic")
    artifacts = {"rules": [{"id": "prefix", "target": "x", "kind": "single",
                             "source": 'def check(row):\n    return row["x"].startswith("b")',
                             "description": "Synthetic invalid prefix"}]}
    one = adapted_zeroed(table, artifacts, max_iter=80).validate(table.shape)
    two = adapted_zeroed(table, artifacts, max_iter=80).validate(table.shape)
    assert np.array_equal(one.probabilities, two.probabilities)
    assert np.all(one.probabilities[:, 1] == 0)
    assert one.repairs == {}
    assert one.metadata["reproduced_upstream"] is False
    assert one.metadata["baseline"] == "adapted_zeroed"
    assert one.metadata["columns"]["x"]["status"] == "fitted"
    assert one.metadata["columns"]["x"]["representatives"] == 5


def test_naive_correction_is_gated_and_does_not_change_detection():
    table = DirtyTable("toy", ("x", "y"), (("old", "old"), ("old", "old")), "synthetic")
    detection = Prediction(np.array([[.9, .1], [.1, .9]]), {}, {"baseline": "adapted_zeroed"})
    result = adapted_zeroed_zeroec(table, detection, {
        "value_repairs": {"x": {"old": "new-x"}, "y": {"old": "new-y"}},
        # Pass-2/loop artifacts are deliberately ignored by the sequential path.
        "dependencies": [{"determinants": ["x"], "dependent": "y"}],
        "pass2": {"repairs": [{"row": 0, "column": "y", "value": "wrong"}]},
    }).validate(table.shape)
    assert result.repairs == {(0, 0): "new-x", (1, 1): "new-y"}
    assert np.array_equal(result.probabilities, detection.probabilities)
    assert result.probabilities is not detection.probabilities
    assert detection.repairs == {}
    assert result.metadata["baseline"] == "adapted_zeroed_zeroec"
    assert result.metadata["reproduced_upstream"] is False


def _source_detection(table, cells):
    probabilities = np.zeros(table.shape)
    for cell in cells:
        probabilities[cell] = 1
    return Prediction(probabilities, {}, {"name": "zeroed_upstream", "input_fingerprint": table.fingerprint})


def test_zeroec_nearest_matches_pinned_langchain_l2_fallback():
    # DOT_PRODUCT in langchain-community==0.0.28 creates IndexFlatL2, so
    # an unnormalized concatenated row must use Euclidean scoring, not cosine.
    vectors = np.array([[1., 1., 0., 0.], [0., 0., 1., 1.], [-1., 1., 0., 0.]], dtype=np.float32)
    query = np.array([.5, .5, 0., 0.], dtype=np.float32)
    indices, scores = nearest_rows(vectors, query)
    brute = ((vectors - query) ** 2).sum(axis=1)
    assert np.array_equal(indices, np.argsort(brute, kind="stable"))
    assert np.allclose(scores, brute[indices])
    # Zero masked queries are valid; resolve equal distances in row order.
    assert nearest_rows(vectors, np.zeros(4, dtype=np.float32))[0].tolist() == [0, 1, 2]


def test_zeroec_mi_filters_singleton_pairs():
    import pandas as pd
    frame = pd.DataFrame({"x": ["a", "a", "b", "b"], "related": ["1", "1", "2", "2"],
                          "unique": ["u", "v", "w", "z"], "constant": ["k"] * 4})
    assert np.allclose(normalized_mutual_information(frame, "x"), [1., 1., 0., 0.])


def test_source_cts_reproduces_prior_retrieval_and_excludes_self_only_when_selected(tmp_path, monkeypatch):
    import zeroclean.upstream_zeroec as zeroec
    table = DirtyTable("toy", ("x", "y"),
                       (("bad", "A"), ("good", "A"), ("good", "A"), ("other", "B"), ("other", "B")),
                       "synthetic")
    matrix = np.array([[[1., 0.], [1., 0.]], [[0., 1.], [1., 0.]],
                       [[0., 1.], [1., 0.]], [[-1., 0.], [0., 1.]],
                       [[-1., 0.], [0., 1.]]], dtype=np.float32)
    encodings = []
    def embeddings(*args):
        encodings.append(1)
        return matrix.copy(), {"synthetic": "fixed-test-embedding"}
    monkeypatch.setattr(zeroec, "_embedding_matrix", embeddings)
    detection = _source_detection(table, [(0, 0)])
    cts = SourceCTS(table, detection, tmp_path)
    retrieved = cts.retrieve(0, "x")
    assert retrieved["indices"] == [1, 2, 0]
    assert retrieved["tuples"] == [{"x": "good", "y": "A"}, {"x": "good", "y": "A"}, {"y": "A"}]
    assert retrieved["context"] == '{"x": "good", "y": "A"}\n{"x": "good", "y": "A"}\n{"y": "A"}\n'
    assert retrieved["related_column_indices"] == [0, 1]
    assert retrieved["weights"] == [1., .7]
    assert retrieved["excluded_query_row"] is False
    assert not cts.matrix.flags.writeable and not cts.detected.flags.writeable
    state = cts.column_state("x")
    assert cts.column_state(0) is state
    # Run the previous standalone source formula independently against this
    # fixed vector fixture to catch any refactoring change in masks/ranking.
    query = matrix[0].copy()
    query[0] = 0.
    query *= np.array([1., .7])[:, None]
    nearest, scores = nearest_rows(matrix.reshape(5, 4), query.flatten())
    prior = [{"index": int(idx), "score": round(score, 2), "target_column": int(idx == 0),
              "sum": int(idx == 0), "page_content": state["documents"][idx]}
             for idx, score in zip(nearest, scores)]
    assert retrieved["indices"] == [item["index"] for item in cts.helpers["sort_dicts"](
        prior, "score", "target_column", "sum")[:3]]
    loop = cts.retrieve(0, 0, exclude_query_row=True)
    assert loop["indices"] == [1, 2, 3]
    assert loop["excluded_query_row"] is True
    assert loop["provenance"]["exclude_query_row"] is True
    assert cts.retrieve(0, 0)["indices"] == [1, 2, 0]
    assert len(encodings) == 1
    assert len(cts.column_states) == 1


def test_source_cts_accepts_uncertain_probabilities_without_positive_detections(tmp_path, monkeypatch):
    import zeroclean.upstream_zeroec as zeroec
    table = DirtyTable("toy", ("x", "y"), (("a", "1"), ("a", "1"), ("b", "2"), ("b", "2")), "synthetic")
    monkeypatch.setattr(zeroec, "_embedding_matrix", lambda *args: (np.ones((4, 2, 2), dtype=np.float32), {}))
    probabilities = np.full(table.shape, .25)
    cts = SourceCTS(table, probabilities, tmp_path, exclude_query_row=True)
    probabilities[:] = 1.  # The CTS instance retains its original detector copy.
    assert not cts.detected.any()
    assert (cts.probabilities == .25).all()
    retrieved = cts.retrieve(0, 0)
    assert retrieved["indices"] == [1, 2, 3]
    assert retrieved["provenance"]["uses_answer_examples"] is False
    with pytest.raises(ValueError, match="row index"):
        cts.retrieve(4, 0)
    with pytest.raises(ValueError, match="column"):
        cts.retrieve(0, "absent")


def test_source_cts_lru_eviction_preserves_complete_context_json(tmp_path, monkeypatch):
    import zeroclean.upstream_zeroec as zeroec
    table = DirtyTable("toy", ("a", "b", "c", "d", "e"),
                       tuple(tuple(f"{column}{row // 2}" for column in "abcde") for row in range(6)),
                       "synthetic")
    matrix = np.random.default_rng(42).normal(size=(6, 5, 3)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=2, keepdims=True)
    monkeypatch.setattr(zeroec, "_embedding_matrix", lambda *args: (matrix.copy(), {"fixture": "fixed"}))
    monkeypatch.setattr(zeroec, "normalized_mutual_information", lambda *args: np.ones(5))
    detection = _source_detection(table, [(0, 0), (2, 1), (4, 2), (1, 3), (3, 4)])
    unbounded = SourceCTS(table, detection, tmp_path / "unbounded")
    bounded = SourceCTS(table, detection, tmp_path / "bounded", max_cached_columns=2)
    first_state = bounded.column_state(0)
    expected_order = []
    for column in (0, 1, 0, 2, 3, 2, 4, 0, 1):
        if column in expected_order:
            expected_order.remove(column)
        expected_order.append(column)
        expected_order = expected_order[-2:]
        for exclude, target_mask in ((False, False), (True, False), (True, True)):
            kwargs = {"exclude_query_row": exclude, "mask_query_target": target_mask}
            actual = bounded.retrieve(column % 6, column, **kwargs)
            expected = unbounded.retrieve(column % 6, column, **kwargs)
            assert json.dumps(actual) == json.dumps(expected)
        assert list(bounded.column_states) == expected_order
        assert len(bounded.column_states) <= 2
    assert len(unbounded.column_states) == 5
    # Column zero was evicted and reconstructed without changing its arrays or
    # touching the immutable shared matrix and detector decisions.
    rebuilt = bounded.column_state(0)
    assert rebuilt is not first_state
    for key in ("indices", "weights", "vectors", "squared_norms", "column_sums"):
        assert np.array_equal(rebuilt[key], first_state[key])
        assert not rebuilt[key].flags.writeable
    assert rebuilt["documents"] == first_state["documents"]
    assert np.array_equal(bounded.matrix, matrix)
    assert bounded.provenance == unbounded.provenance
    assert "max_cached_columns" not in bounded.provenance
    for invalid in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="max_cached_columns"):
            SourceCTS(table, detection, tmp_path, max_cached_columns=invalid)


def test_zeroec_bounded_cache_preserves_requests_and_repairs(tmp_path, monkeypatch):
    import zeroclean.upstream_zeroec as zeroec
    table = DirtyTable("toy", ("a", "b", "c", "d"),
                       tuple(tuple(f"{column}{row // 2}" for column in "abcd") for row in range(6)),
                       "synthetic")
    matrix = np.random.default_rng(19).normal(size=(6, 4, 3)).astype(np.float32)
    monkeypatch.setattr(zeroec, "_embedding_matrix", lambda *args: (matrix.copy(), {}))
    detection = _source_detection(table, [(0, 0), (4, 0), (1, 1), (5, 1), (2, 2), (3, 3)])
    instances = []
    class ObservedCTS(SourceCTS):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.visited_columns = []
            instances.append(self)
        def retrieve(self, row, column, *args, **kwargs):
            self.visited_columns.append(column)
            result = super().retrieve(row, column, *args, **kwargs)
            assert len(self.column_states) <= 2
            return result
    monkeypatch.setattr(zeroec, "SourceCTS", ObservedCTS)
    calls = []
    def request(stage, prompt, metadata):
        calls.append((stage, prompt, metadata))
        return json.dumps({"correction": {metadata["column"]: "fixed"}})
    bounded = run_reconstructed_zeroed_zeroec(table, detection, request, tmp_path / "bounded",
                                              use_general_examples=False)
    bounded_calls = json.dumps(calls)
    assert instances[0].max_cached_columns == 2
    assert instances[0].visited_columns == [0, 0, 1, 1, 2, 3]
    class UnboundedCTS(SourceCTS):
        def __init__(self, *args, **kwargs):
            kwargs["max_cached_columns"] = None
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(zeroec, "SourceCTS", UnboundedCTS)
    calls.clear()
    unbounded = run_reconstructed_zeroed_zeroec(table, detection, request, tmp_path / "unbounded",
                                                use_general_examples=False)
    assert json.dumps(calls) == bounded_calls
    assert bounded.repairs == unbounded.repairs
    assert bounded.metadata == unbounded.metadata
    assert np.array_equal(bounded.probabilities, unbounded.probabilities)


def test_source_cts_optional_target_mask_changes_only_query_target_and_provenance(tmp_path, monkeypatch):
    import zeroclean.upstream_zeroec as zeroec
    table = DirtyTable("toy", ("target", "uncertain_context", "detected_context"),
                       (("q", "A", "bad"), ("x-near", "B", "bad"),
                        ("y-near", "A", "bad"), ("far", "C", "bad")), "synthetic")
    matrix = np.array([[[1., 0.], [1., 0.], [1., 0.]],
                       [[1., 0.], [0., 1.], [0., 1.]],
                       [[-1., 0.], [1., 0.], [0., 1.]],
                       [[-1., 0.], [-1., 0.], [0., 1.]]], dtype=np.float32)
    monkeypatch.setattr(zeroec, "_embedding_matrix", lambda *args: (matrix.copy(), {}))
    monkeypatch.setattr(zeroec, "normalized_mutual_information", lambda *args: np.ones(3))
    probabilities = np.full(table.shape, .25)
    probabilities[:, 2] = .9
    original_probabilities = probabilities.copy()
    cts = SourceCTS(table, probabilities, tmp_path, exclude_query_row=True)
    documents = list(cts.column_state(0)["documents"])
    queries = []
    nearest = zeroec.nearest_rows
    def observe(vectors, query, k=30, squared_norms=None):
        queries.append(query.copy())
        return nearest(vectors, query, k, squared_norms)
    monkeypatch.setattr(zeroec, "nearest_rows", observe)
    baseline = cts.retrieve(0, 0, k=2)
    explicit_false = cts.retrieve(0, 0, k=2, mask_query_target=False)
    masked = cts.retrieve(0, 0, k=2, mask_query_target=True)
    assert json.dumps(baseline) == json.dumps(explicit_false)
    assert "mask_query_target" not in baseline["provenance"]
    assert baseline["indices"] == [1, 2]
    assert masked["indices"] == [2, 1]
    assert queries[0].tolist() == [1., 0., 1., 0., 0., 0.]
    assert queries[-1].tolist() == [0., 0., 1., 0., 0., 0.]
    assert masked["provenance"]["mask_query_target"] is True
    assert documents == cts.column_state(0)["documents"]
    assert all("uncertain_context" in value and "target" in value and "detected_context" not in value
               for value in masked["tuples"])
    assert np.array_equal(probabilities, original_probabilities)
    assert np.array_equal(cts.matrix, matrix)
    assert np.array_equal(cts.detected.astype(bool), original_probabilities > .5)
    assert json.dumps(baseline) == json.dumps(cts.retrieve(0, 0, k=2))


def test_zeroec_exact_preflight_does_not_silently_replace_missing_templates():
    status = preflight_zeroec()
    assert not status["ready"]
    assert "SystemMessage_EC.txt" in "; ".join(status["blockers"])
    assert status["reconstruction"]["fidelity"] == "source_derived_reconstruction"
    assert status["reconstruction"]["reproduced_upstream"] is False


def test_author_prompt_profile_is_explicit_hash_bound_and_zero_example(tmp_path, monkeypatch):
    import zeroclean.upstream_zeroec as zeroec
    source = Path("vendor/ZeroEC")
    status = zeroec.reconstruction_preflight(source, source / "all-MiniLM-L6-v2", False,
                                             zeroec.AUTHOR_PROMPT_PROFILE)
    assert status["ready"]
    assert status["fidelity"] == "author_source_reconstruction"
    assert status["prompt_provenance"]["historical_identity_established"] is False
    assert status["prompt_provenance"]["commit"] == zeroec.AUTHOR_PROMPT_COMMIT
    assert status["strict_zero_shot"] is True
    assert not any("general_examples" in name for name in status["source_sha256"])
    helpers = zeroec._source_helpers(source)
    author_system, author_human, examples = zeroec._prompt_templates(
        source, helpers, False, zeroec.AUTHOR_PROMPT_PROFILE)
    committed_system, committed_human, _ = zeroec._prompt_templates(source, helpers, False)
    assert author_system != committed_system and author_human == committed_human
    assert "{dataset_specific_examples}" in author_system
    assert examples == ""
    table = DirtyTable("toy", ("x", "y"), (("bad", "A"), ("good", "A"), ("good", "A")), "synthetic")
    detection = _source_detection(table, [(0, 0)])
    monkeypatch.setattr(zeroec, "_embedding_matrix", lambda *args: (np.ones((3, 2, 2), dtype=np.float32), {}))
    calls = []
    def request(stage, prompt, metadata):
        calls.append((prompt, metadata))
        return '{"correction":{"x":"good"}}'
    result = run_reconstructed_zeroed_zeroec(table, detection, request, tmp_path,
                                              use_general_examples=False,
                                              prompt_profile=zeroec.AUTHOR_PROMPT_PROFILE)
    assert result.repairs == {(0, 0): "good"}
    assert "avoiding NULL corrections under any circumstance" in calls[0][0]
    assert "1025 W ADDISON SX" not in calls[0][0]
    assert calls[0][1]["prompt_profile"] == zeroec.AUTHOR_PROMPT_PROFILE
    assert calls[0][1]["use_general_examples"] is False
    assert result.metadata["prompt_profile"] == zeroec.AUTHOR_PROMPT_PROFILE
    assert result.metadata["strict_zero_shot"] is True
    assert result.metadata["reproduced_upstream"] is False


@pytest.mark.parametrize("damage", ["prompt", "provenance"])
def test_author_prompt_profile_rejects_tampering_without_fallback(tmp_path, damage):
    import shutil
    from zeroclean.upstream_zeroec import AUTHOR_PROMPT_PROFILE, DEFAULT_RECOVERED_PROMPT_DIR
    recovered = tmp_path / "recovered"
    shutil.copytree(DEFAULT_RECOVERED_PROMPT_DIR, recovered)
    if damage == "prompt":
        with (recovered / "SystemMessage_EC.txt").open("a") as handle:
            handle.write("\nUnauthorized template change")
    else:
        path = recovered / "provenance.json"
        audit = json.loads(path.read_text())
        audit["source_commit"] = "different-revision"
        path.write_text(json.dumps(audit))
    table = DirtyTable("toy", ("x",), (("bad",),), "synthetic")
    calls = []
    with pytest.raises(BaselineUnavailable, match="SHA256 mismatch|Invalid author-recovered"):
        run_reconstructed_zeroed_zeroec(table, _source_detection(table, [(0, 0)]),
                                         lambda *args: calls.append(args), tmp_path / "out",
                                         use_general_examples=False, prompt_profile=AUTHOR_PROMPT_PROFILE,
                                         recovered_prompt_dir=recovered)
    assert calls == []
    assert not (tmp_path / "out" / "run_cache").exists()


def test_unknown_prompt_profile_fails_closed(tmp_path):
    table = DirtyTable("toy", ("x",), (("bad",),), "synthetic")
    with pytest.raises(BaselineUnavailable, match="Unknown ZeroEC prompt profile"):
        run_reconstructed_zeroed_zeroec(table, _source_detection(table, [(0, 0)]),
                                         lambda *args: "", tmp_path, prompt_profile="unspecified-fallback")


def test_zeroec_sequential_reconstruction_rejects_adapted_detector(tmp_path):
    table = DirtyTable("toy", ("x",), (("dirty",),), "synthetic")
    detection = Prediction(np.ones(table.shape), {}, {"baseline": "adapted_zeroed"})
    with pytest.raises(ValueError, match="requires source ZeroED"):
        run_reconstructed_zeroed_zeroec(table, detection, lambda *args: "", tmp_path)


def test_zeroec_sequential_reconstruction_rejects_different_dirty_fingerprint(tmp_path):
    table = DirtyTable("toy", ("x",), (("dirty",),), "synthetic")
    detection = _source_detection(table, [(0, 0)])
    detection.metadata["input_fingerprint"] = "different-input"
    with pytest.raises(ValueError, match="different dirty input"):
        run_reconstructed_zeroed_zeroec(table, detection, lambda *args: "", tmp_path)


def test_upstream_predicates_and_distribution_are_confined():
    from zeroclean.upstream_sandbox import UpstreamPredicateSandbox, distribution_result
    records = [{"x": "A12"}, {"x": "bad"}]
    sandbox = UpstreamPredicateSandbox(records)
    code = 'def valid(value, attr):\n    import re\n    return re.fullmatch(r"A[0-9]+", value[attr])'
    assert sandbox.evaluate(code, records[0], "x") == 1
    assert sandbox.evaluate(code, records[1], "x") == 0
    assert sandbox.rejections == {}
    rejected = 'def invalid(value, attr):\n    import os\n    return os.path.exists(value[attr])'
    assert sandbox.evaluate(rejected, records[0], "x") == -1
    assert sandbox.rejections
    analysis = 'def describe(dirty_csv, attr_name):\n    return "distinct=" + str(dirty_csv[attr_name].nunique())'
    assert distribution_result(analysis, records, "x") == "distinct=2"
    forbidden = 'def describe(dirty_csv, attr_name):\n    return open("/etc/passwd").read()'
    assert distribution_result(forbidden, records, "x").startswith("Error executing function:")


def test_upstream_predicate_cache_fill_is_shared_across_source_workers(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import zeroclean.upstream_sandbox as sandbox_module
    from zeroclean.upstream_sandbox import UpstreamPredicateSandbox
    records = [{"x": "good"}, {"x": "bad"}]
    sandbox = UpstreamPredicateSandbox(records)
    calls = []
    ready = threading.Barrier(16)
    def evaluate_batch(rule, rows):
        calls.append(len(rows))
        threading.Event().wait(.02)  # Let every source worker reach the miss.
        return [int(row["x"] == "good") for row in rows], {"status": "ok", "reason": "", "error_samples": []}
    monkeypatch.setattr(sandbox_module, "_predicate_batch", evaluate_batch)
    code = 'def valid(value, attr):\n    return value[attr] == "good"'
    def source_worker(i):
        ready.wait()
        return sandbox.evaluate(code, records[i % 2], "x")
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(source_worker, range(16)))
    assert results == [1, 0] * 8
    assert calls == [2]
    # The same lock permits an incremental extension; old rows are not rerun.
    assert sandbox.evaluate(code, {"x": "new-synthetic-value"}, "x") == 0
    assert calls == [2, 1]


def test_upstream_cache_primes_main_and_feature_numeric_representations(tmp_path, monkeypatch):
    import zeroclean.upstream_sandbox as sandbox_module
    from zeroclean.upstream_zeroed import _source_dirty_frames
    from zeroclean.upstream_sandbox import UpstreamPredicateSandbox
    path = tmp_path / "dirty-only.csv"
    path.write_text("number,context\n12,A\n13.5,B\n,A\n")
    main, feature = _source_dirty_frames(path)
    assert main.loc[0, "number"] == "12"
    assert feature.loc[0, "number"] == "12.0"
    calls = []
    def evaluate_batch(rule, rows):
        calls.append(list(rows))
        return [int(row["number"] == "12") for row in rows], {"status": "ok", "reason": "", "error_samples": []}
    monkeypatch.setattr(sandbox_module, "_predicate_batch", evaluate_batch)
    sandbox = UpstreamPredicateSandbox(main.to_dict("records"))
    sandbox.add_records(feature.to_dict("records"))
    code = 'def valid(value, attr):\n    return value[attr] == "12"'
    assert sandbox.evaluate(code, main.iloc[0].to_dict(), "number") == 1
    assert sandbox.evaluate(code, feature.iloc[0].to_dict(), "number") == 0
    for _ in range(5):
        for value in main.to_dict("records") + feature.to_dict("records"):
            sandbox.evaluate(code, value, "number")
    assert len(calls) == 1 and len(calls[0]) == 4
    assert {row["number"] for row in calls[0]} == {"12", "12.0", "13.5", "nan"}
    assert sandbox.evaluation_batches == 1


def test_upstream_cache_extension_evaluates_only_pending_new_rows(monkeypatch):
    import zeroclean.upstream_sandbox as sandbox_module
    from zeroclean.upstream_sandbox import UpstreamPredicateSandbox
    calls = []
    def evaluate_batch(rule, rows):
        calls.append([row["x"] for row in rows])
        return [int(row["x"].isdigit()) for row in rows], {"status": "ok", "reason": "", "error_samples": []}
    monkeypatch.setattr(sandbox_module, "_predicate_batch", evaluate_batch)
    sandbox = UpstreamPredicateSandbox([{"x": str(i)} for i in range(100)])
    code = 'def valid(value, attr):\n    return value[attr].isdigit()'
    assert sandbox.evaluate(code, {"x": "12"}, "x") == 1
    for i in range(20):
        assert sandbox.evaluate(code, {"x": f"new-{i}"}, "x") == 0
    assert [len(batch) for batch in calls] == [100] + [1] * 20
    sandbox.add_records([{"x": "batched-a"}, {"x": "batched-b"}])
    assert sandbox.evaluate(code, {"x": "batched-a"}, "x") == 0
    assert sandbox.evaluate(code, {"x": "batched-b"}, "x") == 0
    assert calls[-1] == ["batched-a", "batched-b"]
    assert sandbox.evaluated_rows == 122


def test_supported_upstream_runtime_errors_are_preserved_without_policy_rejection():
    from zeroclean.upstream_sandbox import UpstreamPredicateSandbox
    sandbox = UpstreamPredicateSandbox([{"x": "12"}, {"x": "13"}])
    code = 'def valid(value, attr):\n    return value[attr].isdigit()'
    # Source feature.py sometimes supplies an outer attr missing from this
    # related tuple. Native handle_func_exec catches KeyError and emits -1.
    assert sandbox.evaluate(code, {"x": "12"}, "outer_attr") == -1
    assert sandbox.evaluate(code, {"x": "13"}, "outer_attr") == -1
    assert sandbox.rejections == {}
    assert sandbox.evaluation_errors == 2
    assert len(sandbox.runtime_errors) == 1
    assert "KeyError" in sandbox.runtime_errors[0]["reason"]
    # A predicate that does not inspect attr may still succeed on that tuple.
    ignores_attr = 'def valid(value, attr):\n    return value["x"].isdigit()'
    assert sandbox.evaluate(ignores_attr, {"x": "12"}, "outer_attr") == 1
    assert sandbox.rejections == {}


def test_source_metadata_target_override_preserves_default_validation():
    from zeroclean.sandbox import _execute
    rule = {"id": "source-target", "target": "missing", "kind": "single", "description": "Synthetic source adapter",
            "source": 'def check(row):\n    return row["x"].isdigit()'}
    values, report = _execute([rule], [{"x": "12"}])
    assert values == [[-1]] and report[0]["status"] == "invalid_rule"
    values, report = _execute([rule], [{"x": "12"}], require_target=False)
    assert values == [[1]] and report[0]["status"] == "ok"
    rule["source"] = 'def check(row):\n    return open("/etc/passwd").read() == ""'
    values, report = _execute([rule], [{"x": "12"}], require_target=False)
    assert values == [[-1]] and report[0]["status"] == "invalid_rule"


def test_old_clustering_cache_retained_only_when_predicate_geometry_matches():
    import pandas as pd
    from zeroclean.upstream_zeroed import _refresh_cached_predicate_features
    frame = pd.DataFrame({"x": ["a", "b"]})
    related = {"x": {}}
    preliminary = {"x": {"clean": ["synthetic-predicate"]}}
    class Evaluator:
        def __init__(self, outputs): self.outputs = outputs
        def evaluate(self, code, row, attr): return self.outputs[row["x"]]
    old = {"features": {(0, 0): {"pre_funcs_feat": np.array([-1])},
                         (1, 0): {"pre_funcs_feat": np.array([-1])}}}
    report = _refresh_cached_predicate_features(old, frame, ["x"], related, preliminary, Evaluator({"a": 1, "b": 1}))
    assert report["compatible"] and report["raw_predicates_changed"]
    assert old["features"][(0, 0)]["pre_funcs_feat"].tolist() == [1]
    changed = {"features": {(0, 0): {"pre_funcs_feat": np.array([0])},
                             (1, 0): {"pre_funcs_feat": np.array([1])}}}
    report = _refresh_cached_predicate_features(changed, frame, ["x"], related, preliminary,
                                                Evaluator({"a": 1, "b": 0}))
    assert not report["compatible"] and "geometry changed" in report["reason"]


def test_zeroec_real_cpu_embedding_is_normalized():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from zeroclean.upstream_zeroec import MiniLMCPUEncoder
    root = Path("vendor/ZeroEC/all-MiniLM-L6-v2")
    if not (root / "quantized_true.onnx").is_file():
        pytest.skip("Bundled upstream embedding graph unavailable")
    encoder = MiniLMCPUEncoder(root)
    vectors = encoder.encode(["a small city", "a large town", "1900"])
    assert encoder.max_length == 512  # pinned FastSentenceTransformer override
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]


def test_embedding_cache_serializes_processes_and_keeps_existing_memmaps(tmp_path, monkeypatch):
    import multiprocessing
    import pandas as pd
    import zeroclean.upstream_zeroec as zeroec
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("Process cache concurrency test requires fork")
    context = multiprocessing.get_context("fork")
    embedding = tmp_path / "model"
    embedding.mkdir()
    for name in zeroec.MODEL_FILES:
        path = embedding / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"synthetic-model-resource")
    calls = context.Value("i", 0)
    class Encoder:
        def __init__(self, root): pass
        def encode(self, texts):
            with calls.get_lock(): calls.value += 1
            # Long enough for every independent process to contend for cache
            # creation; output is tiny and has no ONNX/BLAS dependency.
            import time
            time.sleep(.05)
            return np.asarray([[len(text), sum(map(ord, text)), 1] for text in texts], dtype=np.float32)
    monkeypatch.setattr(zeroec, "MiniLMCPUEncoder", Encoder)
    table = DirtyTable("toy", ("x", "y"), (("a", "1"), ("b", "2")), "synthetic-v1")
    frame = pd.DataFrame(table.rows, columns=table.columns)
    output = tmp_path / "cache"
    start, results = context.Barrier(4), context.Queue()
    def worker():
        start.wait(timeout=10)
        matrix, _ = zeroec._embedding_matrix(table, frame, embedding, output)
        results.put((matrix.tolist(), matrix.flags.writeable, str(matrix.filename)))
        matrix._mmap.close()
    workers = [context.Process(target=worker) for _ in range(4)]
    for process in workers: process.start()
    returned = [results.get(timeout=15) for _ in workers]
    for process in workers:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert calls.value == 1
    assert all(item == returned[0] for item in returned)
    assert returned[0][1] is False
    held, _ = zeroec._embedding_matrix(table, frame, embedding, output)
    original = held.copy()
    changed = DirtyTable("toy", table.columns, (("longer", "1"), ("different", "2")), "synthetic-v2")
    changed_frame = pd.DataFrame(changed.rows, columns=changed.columns)
    newer, _ = zeroec._embedding_matrix(changed, changed_frame, embedding, output)
    assert calls.value == 2
    assert np.array_equal(held, original)
    assert str(held.filename) != str(newer.filename)
    assert not np.array_equal(held, newer)
    metadata = json.loads((output / "run_cache" / "embedding_key.json").read_text())
    assert Path(newer.filename).name == metadata["matrix_file"]
    assert not (Path(newer.filename).stat().st_mode & 0o222)
    assert not list((output / "run_cache").glob("*.tmp"))
    held._mmap.close()
    newer._mmap.close()


def test_embedding_cache_recovers_publication_before_metadata_commit(tmp_path, monkeypatch):
    import pandas as pd
    import zeroclean.types as types
    import zeroclean.upstream_zeroec as zeroec
    embedding = tmp_path / "model"
    embedding.mkdir()
    for name in zeroec.MODEL_FILES:
        path = embedding / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"synthetic-model-resource")
    calls = []
    class Encoder:
        def __init__(self, root): pass
        def encode(self, texts):
            calls.append(1)
            return np.ones((len(texts), 3), dtype=np.float32)
    monkeypatch.setattr(zeroec, "MiniLMCPUEncoder", Encoder)
    table = DirtyTable("toy", ("x",), (("a",), ("b",)), "synthetic")
    frame = pd.DataFrame(table.rows, columns=table.columns)
    output = tmp_path / "cache"
    with monkeypatch.context() as patch:
        def fail_commit(*args): raise OSError("synthetic interruption before metadata commit")
        patch.setattr(types, "write_json", fail_commit)
        with pytest.raises(OSError, match="synthetic interruption"):
            zeroec._embedding_matrix(table, frame, embedding, output)
    assert len(calls) == 1
    assert not (output / "run_cache" / "embedding_key.json").exists()
    assert len(list((output / "run_cache").glob("embeddings.*.npy"))) == 1
    matrix, _ = zeroec._embedding_matrix(table, frame, embedding, output)
    assert len(calls) == 1 and matrix.shape == (2, 1, 3)
    assert (output / "run_cache" / "embedding_key.json").is_file()
    assert not list((output / "run_cache").glob("*.tmp"))
    matrix._mmap.close()


def test_zeroec_queues_every_detected_cell_and_replays_without_changing_detection(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from zeroclean.llm import FileLLMQueue, ResponsePending
    from zeroclean.types import digest
    table = DirtyTable("toy", ("x", "y"),
                       (("bad", "A"), ("good", "A"), ("good", "A"), ("other", "B"), ("other", "B")),
                       "synthetic")
    detection = _source_detection(table, [(0, 0), (3, 1)])
    # Prior detector repairs must never be used as sequential correction input.
    detection.repairs[(1, 0)] = "forbidden feedback"
    queue = FileLLMQueue(tmp_path / "queue")
    out = tmp_path / "run"
    with pytest.raises(ResponsePending, match="all detected cells were enqueued"):
        run_reconstructed_zeroed_zeroec(table, detection, queue.request, out,
                                         use_general_examples=False)
    requests = list((tmp_path / "queue" / "requests").glob("*.json"))
    assert len(requests) == 2
    for path in requests:
        body = json.loads(path.read_text())
        metadata = body["metadata"]
        assert metadata["reproduced_upstream"] is False
        assert metadata["fidelity"] == "source_derived_reconstruction"
        assert "forbidden feedback" not in body["prompt"]
        column = metadata["column"]
        response = {"request_hash": digest(body), "model": queue.model,
                    "reasoning_effort": queue.effort,
                    "response": json.dumps({"correction": {column: "fixed-" + column}})}
        response_path = tmp_path / "queue" / "responses" / path.name
        response_path.parent.mkdir(exist_ok=True)
        response_path.write_text(json.dumps(response))
    result = run_reconstructed_zeroed_zeroec(table, detection, queue.request, out,
                                              use_general_examples=False)
    assert result.repairs == {(0, 0): "fixed-x", (3, 1): "fixed-y"}
    assert np.array_equal(result.probabilities, detection.probabilities)
    assert result.probabilities is not detection.probabilities
    assert result.metadata["strict_zero_shot"] is True
    assert result.metadata["uses_ground_truth"] is False
    assert result.metadata["completed_cells"] == 2
    assert result.metadata["pending_cells"] == 0
    assert result.metadata["name"] == "zeroed_zeroec_reconstruction"
    assert Prediction.load(out / "prediction.json").repairs == result.repairs


def test_zeroec_imdb_context_disabled_and_invalid_replies_have_distinct_retries(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    table = DirtyTable("imdb", ("x", "y"), (("bad", "A"), ("good", "A"), ("good", "A")), "synthetic")
    detection = _source_detection(table, [(0, 0)])
    calls = []
    def request(stage, prompt, metadata):
        calls.append((stage, prompt, metadata))
        return "invalid JSON" if metadata["attempt"] == 0 else '{"correction": {"x": null}}'
    result = run_reconstructed_zeroed_zeroec(table, detection, request, tmp_path,
                                              use_general_examples=True)
    assert [call[2]["attempt"] for call in calls] == [0, 1]
    assert all(call[2]["retrieved_row_indices"] == [] for call in calls)
    assert "Relevant Clean Tuple:  \n" in calls[0][1]
    assert result.repairs == {(0, 0): ""}
    assert result.metadata["strict_zero_shot"] is False
    assert len(result.metadata["response_errors"]) == 1
    assert "empty CSV cell" in result.metadata["null_sentinel_policy"]


def test_zeroec_uses_saved_source_mask_with_exact_half_probability(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    table = DirtyTable("toy", ("x", "y"), (("bad", "A"), ("good", "A"), ("good", "A")), "synthetic")
    probabilities = np.array([[.85, .5], [.25, .1], [.2, .3]])
    detection = Prediction(probabilities, {}, {"name": "zeroed_upstream",
                           "upstream_binary_decisions": (probabilities > .5).tolist(),
                           "decision_threshold": .5, "decision_threshold_operator": ">"})
    calls = []
    def request(stage, prompt, metadata):
        calls.append(metadata)
        return '{"correction":{"x":"good"}}'
    result = run_reconstructed_zeroed_zeroec(table, detection, request, tmp_path,
                                              use_general_examples=False)
    assert len(calls) == 1 and calls[0]["row"] == 0 and calls[0]["column"] == "x"
    assert result.repairs == {(0, 0): "good"}
    assert np.array_equal(result.probabilities, probabilities)
    assert result.metadata["upstream_binary_decisions"] == (probabilities > .5).tolist()
    assert result.metadata["decision_threshold"] == .5
    assert result.metadata["decision_threshold_operator"] == ">"
