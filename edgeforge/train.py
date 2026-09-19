"""Trains a small classical (scikit-learn) model from a CSV file -- the automated version of
TRAINING_A_MODEL.md's own worked example (examples/train_room_comfort.py), for anyone who'd
rather not write that script by hand. Always wraps the chosen estimator in
`Pipeline(StandardScaler(), estimator)`, matching that guide's own strong recommendation, so
EdgeForge folds the rescaling into the generated C automatically (see
edgeforge/ingest/sklearn_ingest.py's Pipeline handling) instead of the raw sensor reading
needing to be pre-scaled by hand.

MODEL_TYPES is deliberately restricted to the five estimator classes
edgeforge/ingest/sklearn_ingest.py's `_SUPPORTED_ESTIMATOR_NAMES` actually accepts -- not
every model type TRAINING_A_MODEL.md's prose mentions in passing (plain `LinearRegression`,
for instance, reads as a natural regression option there but isn't in that supported list, so
training one here would produce a .pkl that fails to ingest). Confirmed against the real list,
not assumed from the guide's own wording.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

from edgeforge.errors import EdgeForgeError

# name -> (task, estimator factory). Hyperparameters match TRAINING_A_MODEL.md's own suggestions
# (MLPClassifier(hidden_layer_sizes=(8,)), a depth-limited tree so it stays small on tiny
# boards) with a higher max_iter than sklearn's own default -- this trains on whatever CSV a
# user uploads, not just the one curated example, and needs to converge reliably on messier
# real-world data without emitting a ConvergenceWarning on every other upload.
MODEL_TYPES: dict[str, tuple[str, "callable"]] = {
    "logistic_regression": ("classification", lambda: LogisticRegression(max_iter=1000)),
    "decision_tree_classifier": ("classification", lambda: DecisionTreeClassifier(max_depth=6, random_state=0)),
    "mlp_classifier": ("classification", lambda: MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=0)),
    "decision_tree_regressor": ("regression", lambda: DecisionTreeRegressor(max_depth=6, random_state=0)),
    "mlp_regressor": ("regression", lambda: MLPRegressor(hidden_layer_sizes=(8,), max_iter=2000, random_state=0)),
}

MODEL_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression (classification) — simple, fast, a solid default first try",
    "decision_tree_classifier": "Decision Tree (classification) — tiniest footprint, fits even 8-bit boards",
    "mlp_classifier": "Small neural network / MLP (classification) — for patterns the two above can't capture",
    "decision_tree_regressor": "Decision Tree (regression) — predicting a number, tiniest footprint",
    "mlp_regressor": "Small neural network / MLP (regression) — predicting a number, more complex patterns",
}

MIN_ROWS_NO_WARNING = 30
MIN_EXAMPLES_PER_LABEL_NO_WARNING = 5


class TrainingDataError(EdgeForgeError):
    """The uploaded CSV, label column, or model choice can't be used to train a model."""


@dataclass
class TrainingReport:
    model_type: str
    task: str
    label_column: str
    feature_columns: list[str]
    n_rows: int
    n_train: int
    n_test: int
    score: float
    label_counts: Optional[dict] = None  # class -> count, classification only
    warnings: list[str] = field(default_factory=list)


def train_from_csv(
    csv_path: Path,
    label_column: str,
    model_type: str,
    out_path: Path,
    test_size: float = 0.2,
    seed: int = 0,
) -> TrainingReport:
    if model_type not in MODEL_TYPES:
        raise TrainingDataError(f"unknown model type {model_type!r} -- choose one of: {', '.join(MODEL_TYPES)}")
    task, make_estimator = MODEL_TYPES[model_type]

    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        raise TrainingDataError(f"couldn't read '{csv_path.name}' as a CSV: {e}") from e

    label_column = label_column.strip()
    if label_column not in data.columns:
        raise TrainingDataError(
            f"column '{label_column}' not found in the CSV. Available columns: {', '.join(data.columns)}"
        )
    feature_columns = [c for c in data.columns if c != label_column]
    if not feature_columns:
        raise TrainingDataError("the CSV needs at least one feature column besides the label column.")

    if data[feature_columns].isnull().any().any() or data[label_column].isnull().any():
        raise TrainingDataError(
            "the CSV has blank cells -- every row needs every column filled in. Fix the data and re-upload."
        )

    try:
        X = data[feature_columns].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TrainingDataError(
            f"every feature column must be numeric -- couldn't convert one of {feature_columns} to numbers ({e})."
        ) from e

    y_raw = data[label_column]
    y = y_raw.to_numpy() if task == "regression" else y_raw.astype(str).to_numpy()

    warnings: list[str] = []
    n_rows = len(data)
    if n_rows < MIN_ROWS_NO_WARNING:
        warnings.append(
            f"only {n_rows} rows — aim for at least a few dozen examples for a model that "
            "generalizes to new data, not just this training set."
        )

    label_counts = None
    stratify = None
    if task == "classification":
        values, counts = np.unique(y, return_counts=True)
        label_counts = dict(zip(values.tolist(), counts.tolist()))
        if len(values) < 2:
            raise TrainingDataError(
                f"column '{label_column}' has only one distinct value ({values[0]!r}) -- a "
                "classifier needs at least two different labels to learn a rule from."
            )
        min_count = int(counts.min())
        if min_count < MIN_EXAMPLES_PER_LABEL_NO_WARNING:
            worst_label = values[counts.argmin()]
            warnings.append(
                f"label {worst_label!r} has only {min_count} example(s) — very imbalanced "
                "labels teach a model to just guess the common one."
            )
        if min_count >= 2:  # stratify needs at least 2 of every class
            stratify = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify
        )
    except ValueError as e:
        raise TrainingDataError(f"couldn't split the data into train/test sets: {e}") from e

    model = Pipeline([("scaler", StandardScaler()), ("estimator", make_estimator())])
    model.fit(X_train, y_train)
    score = float(model.score(X_test, y_test))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    return TrainingReport(
        model_type=model_type,
        task=task,
        label_column=label_column,
        feature_columns=feature_columns,
        n_rows=n_rows,
        n_train=len(X_train),
        n_test=len(X_test),
        score=score,
        label_counts=label_counts,
        warnings=warnings,
    )
