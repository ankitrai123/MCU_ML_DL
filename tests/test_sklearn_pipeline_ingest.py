"""Pipeline(StandardScaler|MinMaxScaler, <estimator>) ingest: the scaler must be folded
into a leading "affine" IR node (never silently dropped), reference_fn must match the
*whole* pipeline's predictions on raw (unscaled) input, and unsupported Pipeline shapes
must fail with a clear message instead of guessing.
"""

import pickle

import numpy as np
import pytest

from edgeforge.boards.registry import BoardRegistry
from edgeforge.codegen.generator import generate
from edgeforge.errors import UnsupportedModelError
from edgeforge.ingest import sklearn_ingest
from edgeforge.quantize.footprint import check_budget
from edgeforge.validate.golden import run_golden_validation


def test_scaled_logreg_ingest_structure(scaled_logreg_path, boards_dir):
    result = sklearn_ingest.ingest(scaled_logreg_path)
    ir = result.ir
    assert ir.kind == "classical"
    assert ir.task == "classification"
    assert ir.input_spec.shape == (4,)
    assert ir.nodes[0].op == "affine"
    assert ir.nodes[0].inputs == ["input"]
    assert ir.nodes[1].op == "linear"
    assert ir.nodes[1].inputs == ir.nodes[0].outputs  # rewired onto the scaled buffer, not raw input
    scale = ir.tensors[ir.nodes[0].attrs["scale"]]
    shift = ir.tensors[ir.nodes[0].attrs["shift"]]
    assert scale.shape == shift.shape == (4,)
    assert ir.weight_bytes() > 0

    board = BoardRegistry(boards_dir).get("stm32f411")
    report = check_budget(ir, board)  # must not raise; also proves footprint accounts for the new tensors
    assert report.param_count > 0


def test_scaled_tree_ingest_structure(scaled_tree_path):
    result = sklearn_ingest.ingest(scaled_tree_path)
    ir = result.ir
    assert ir.nodes[0].op == "affine"
    assert ir.nodes[1].op == "tree"
    assert ir.nodes[1].inputs == ["scaler0_out"]


def test_scaled_logreg_reference_fn_matches_pipeline(scaled_logreg_path):
    with open(scaled_logreg_path, "rb") as f:
        pipeline = pickle.load(f)
    result = sklearn_ingest.ingest(scaled_logreg_path)

    # Raw (unscaled) iris-scale feature values -- not the standardized -3..3 range the bare
    # (unscaled) ingest path assumes -- since a raw sensor-style reading is the real input now.
    raw_samples = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [6.7, 3.1, 4.7, 1.5],
        [7.7, 3.8, 6.7, 2.2],
    ])
    for s in raw_samples:
        cls, out = result.reference_fn(s)
        proba = pipeline.predict_proba(s.reshape(1, -1))[0]
        assert cls == int(np.argmax(proba))
        np.testing.assert_allclose(out, proba, atol=1e-6)


def test_golden_validation_passes_scaled_logreg_on_stm32(scaled_logreg_path, boards_dir, tmp_path):
    reg = BoardRegistry(boards_dir)
    # No --sample-range override: exercises the scaler-derived raw-domain default sampler
    # (mean +/- 3*std from the fitted StandardScaler), not a generic -3..3 guess.
    result = sklearn_ingest.ingest(scaled_logreg_path)
    board = reg.get("stm32f411")
    gen = generate(result.ir, board, tmp_path)
    report = run_golden_validation(result, board, gen.model_c, tmp_path, n_samples=25)
    assert report.all_passed, report.describe()


def test_golden_validation_passes_scaled_tree_on_8051(scaled_tree_path, boards_dir, tmp_path):
    reg = BoardRegistry(boards_dir)
    # Default sampler here comes from MinMaxScaler's data_min_/data_max_ -- the true raw
    # training-data range.
    result = sklearn_ingest.ingest(scaled_tree_path)
    board = reg.get("8051_at89s52")
    gen = generate(result.ir, board, tmp_path)
    report = run_golden_validation(result, board, gen.model_c, tmp_path, n_samples=25)
    assert report.all_passed, report.describe()


def test_golden_validation_detects_corrupted_affine_constant(scaled_logreg_path, boards_dir, tmp_path):
    """A meta-test: corrupt the folded scaler's C arithmetic after generation and confirm
    validation catches it, so a passing suite isn't just because nothing is actually checked."""
    reg = BoardRegistry(boards_dir)
    result = sklearn_ingest.ingest(scaled_logreg_path)
    board = reg.get("stm32f411")
    gen = generate(result.ir, board, tmp_path)

    text = gen.model_c.read_text()
    corrupted = text.replace("+ ef_scaler0_shift", "- ef_scaler0_shift", 1)
    assert corrupted != text
    gen.model_c.write_text(corrupted)

    report = run_golden_validation(result, board, gen.model_c, tmp_path, n_samples=25)
    assert not report.all_passed


def test_pipeline_wrong_step_count_raises(tmp_path, iris_data):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = iris_data
    model = Pipeline([
        ("scaler", StandardScaler()), ("pca", PCA(n_components=2)), ("clf", LogisticRegression()),
    ]).fit(X, y)
    path = tmp_path / "three_step.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    with pytest.raises(UnsupportedModelError, match="3 step"):
        sklearn_ingest.ingest(path)


def test_pipeline_unsupported_preprocessing_raises(tmp_path, iris_data):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X, y = iris_data
    model = Pipeline([("pca", PCA(n_components=2)), ("clf", LogisticRegression())]).fit(X, y)
    path = tmp_path / "pca_pipeline.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    with pytest.raises(UnsupportedModelError, match="not a supported scaler"):
        sklearn_ingest.ingest(path)


def test_pipeline_unsupported_final_estimator_raises(tmp_path, iris_data):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y = iris_data
    model = Pipeline([("scaler", StandardScaler()), ("clf", SVC())]).fit(X, y)
    path = tmp_path / "svc_pipeline.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)

    with pytest.raises(UnsupportedModelError, match="unsupported estimator type"):
        sklearn_ingest.ingest(path)
