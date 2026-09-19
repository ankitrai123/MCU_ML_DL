"""train_from_csv() (Phase 4, "training in the browser"): the automated version of
TRAINING_A_MODEL.md's worked example. MODEL_TYPES is deliberately restricted to the five
estimator classes edgeforge/ingest/sklearn_ingest.py actually accepts -- excluding
LinearRegression, which the guide's prose mentions but which isn't in that supported list --
so every model these tests produce must round-trip cleanly through sklearn_ingest.ingest().
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edgeforge.ingest.sklearn_ingest import ingest
from edgeforge.train import MODEL_TYPES, TrainingDataError, train_from_csv

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def room_comfort_csv() -> Path:
    return REPO_ROOT / "examples" / "room_comfort.csv"


@pytest.fixture(scope="session")
def regression_csv(tmp_path_factory) -> Path:
    rng = np.random.default_rng(0)
    n = 100
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-5, 5, n)
    y = 2.0 * x1 - 0.5 * x2 + rng.normal(0, 0.1, n)
    path = tmp_path_factory.mktemp("data") / "synthetic_regression.csv"
    pd.DataFrame({"x1": x1, "x2": x2, "target": y}).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("model_type", list(MODEL_TYPES))
def test_train_from_csv_round_trips_through_ingest(model_type, room_comfort_csv, regression_csv, tmp_path):
    task, _ = MODEL_TYPES[model_type]
    csv_path = room_comfort_csv if task == "classification" else regression_csv
    label_column = "comfort" if task == "classification" else "target"

    out_path = tmp_path / "model.pkl"
    report = train_from_csv(csv_path, label_column, model_type, out_path)

    assert report.model_type == model_type
    assert report.task == task
    assert report.label_column == label_column
    assert label_column not in report.feature_columns
    assert report.n_train + report.n_test == report.n_rows
    assert 0.0 <= report.score if task == "regression" else 0.0 <= report.score <= 1.0

    result = ingest(out_path)
    ir = result.ir
    assert ir.kind == "classical"
    assert ir.task == task
    assert ir.nodes[0].op == "affine"  # the StandardScaler folded in, never dropped


def test_classification_report_has_label_counts(room_comfort_csv, tmp_path):
    report = train_from_csv(room_comfort_csv, "comfort", "logistic_regression", tmp_path / "m.pkl")
    assert report.label_counts
    assert sum(report.label_counts.values()) == report.n_rows


def test_regression_report_has_no_label_counts(regression_csv, tmp_path):
    report = train_from_csv(regression_csv, "target", "decision_tree_regressor", tmp_path / "m.pkl")
    assert report.label_counts is None


def test_unknown_model_type_raises(room_comfort_csv, tmp_path):
    with pytest.raises(TrainingDataError, match="unknown model type"):
        train_from_csv(room_comfort_csv, "comfort", "linear_regression", tmp_path / "m.pkl")


def test_missing_label_column_raises(room_comfort_csv, tmp_path):
    with pytest.raises(TrainingDataError, match="not found in the CSV"):
        train_from_csv(room_comfort_csv, "nope", "logistic_regression", tmp_path / "m.pkl")


def test_blank_cells_raise(tmp_path):
    csv_path = tmp_path / "blanks.csv"
    csv_path.write_text("a,b,label\n1,2,x\n3,,y\n")
    with pytest.raises(TrainingDataError, match="blank cells"):
        train_from_csv(csv_path, "label", "logistic_regression", tmp_path / "m.pkl")


def test_non_numeric_feature_raises(tmp_path):
    csv_path = tmp_path / "text_feature.csv"
    csv_path.write_text("a,b,label\nfoo,2,x\nbar,4,y\nbaz,6,x\n")
    with pytest.raises(TrainingDataError, match="must be numeric"):
        train_from_csv(csv_path, "label", "logistic_regression", tmp_path / "m.pkl")


def test_single_class_label_raises(tmp_path):
    csv_path = tmp_path / "one_class.csv"
    csv_path.write_text("a,b,label\n1,2,x\n3,4,x\n5,6,x\n")
    with pytest.raises(TrainingDataError, match="only one distinct value"):
        train_from_csv(csv_path, "label", "logistic_regression", tmp_path / "m.pkl")


def test_no_feature_columns_raises(tmp_path):
    csv_path = tmp_path / "label_only.csv"
    csv_path.write_text("label\nx\ny\n")
    with pytest.raises(TrainingDataError, match="at least one feature column"):
        train_from_csv(csv_path, "label", "logistic_regression", tmp_path / "m.pkl")


def test_small_dataset_warns(tmp_path):
    rng = np.random.default_rng(0)
    n = 10
    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame({
        "a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n),
        "label": ["x", "y"] * (n // 2),
    }).to_csv(csv_path, index=False)
    report = train_from_csv(csv_path, "label", "logistic_regression", tmp_path / "m.pkl")
    assert any("rows" in w for w in report.warnings)


def test_imbalanced_labels_warn(tmp_path):
    csv_path = tmp_path / "imbalanced.csv"
    rows = ["a,b,label"]
    for i in range(40):
        rows.append(f"{i},{i * 2},common")
    for i in range(2):
        rows.append(f"{i + 100},{i},rare")
    csv_path.write_text("\n".join(rows) + "\n")
    report = train_from_csv(csv_path, "label", "decision_tree_classifier", tmp_path / "m.pkl")
    assert any("imbalanced" in w or "rare" in w for w in report.warnings)


def test_pipeline_scaler_step_folds_into_affine(room_comfort_csv, tmp_path):
    """Confirms the deliberate Pipeline(StandardScaler, estimator) shape -- step names
    ("scaler"/"estimator") are irrelevant to sklearn_ingest.py, only the step *types* are
    checked, but the scaler must always be present so raw sensor input gets normalized
    on-device the same way training data was."""
    out_path = tmp_path / "m.pkl"
    train_from_csv(room_comfort_csv, "comfort", "mlp_classifier", out_path)

    import pickle

    with open(out_path, "rb") as f:
        model = pickle.load(f)
    assert type(model).__name__ == "Pipeline"
    step_types = [type(step).__name__ for _, step in model.steps]
    assert step_types == ["StandardScaler", "MLPClassifier"]
