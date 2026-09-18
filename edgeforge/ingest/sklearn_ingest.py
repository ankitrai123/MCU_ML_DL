"""scikit-learn pickle -> ModelIR.

Supported estimators: DecisionTreeClassifier/Regressor, LogisticRegression,
MLPClassifier/Regressor -- either a bare fitted estimator, or one wrapped in a
two-step Pipeline(StandardScaler|MinMaxScaler, <one of the above>). A Pipeline
shaped like that has its scaler folded into the generated C as a leading
"affine" IR node (see edgeforge.ir's op vocabulary), so a raw, unscaled sensor
reading gets normalized on-device exactly the way training data was, instead
of the generated code silently expecting an already-scaled input. See
UnsupportedModelError below for what other Pipeline shapes still need instead:
extract the final estimator and fold that preprocessing into training by hand.

Hidden-layer activations are limited to relu/relu6/identity: anything
requiring a transcendental function (sigmoid/tanh/logistic) would need a math
library, which EdgeForge avoids linking so that even the 8051 tier stays
freestanding. The final classification activation (softmax/logistic) is
always safe to skip outright, quantized or not: argmax is invariant under any
monotonic transform, so the raw pre-activation scores decide the same class.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from edgeforge.errors import UnsupportedModelError, UnsupportedOpError
from edgeforge.ingest.base import IngestResult, uniform_range_sampler
from edgeforge.ir import ModelIR, Node, TensorSpec

_SAFE_HIDDEN_ACTIVATIONS = {"relu", "identity"}
_DEFAULT_SAMPLE_RANGE = (-3.0, 3.0)  # plausible range for standardized sklearn features
_SUPPORTED_SCALER_NAMES = ("StandardScaler", "MinMaxScaler")
_SUPPORTED_ESTIMATOR_NAMES = (
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "LogisticRegression", "MLPClassifier", "MLPRegressor",
)


def can_handle(path: Path) -> bool:
    return path.suffix.lower() in (".pkl", ".pickle")


def _load(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301 -- pickle is the documented sklearn interchange format


def ingest(path: Path, sample_range: tuple[float, float] | None = None) -> IngestResult:
    model = _load(path)
    module_name = type(model).__module__

    if module_name.startswith("sklearn.pipeline"):
        return _ingest_pipeline(path, model, sample_range)

    return _ingest_bare_estimator(path, model, type(model).__name__, module_name, sample_range)


def _ingest_bare_estimator(path: Path, model: Any, cls_name: str, module_name: str, sample_range) -> IngestResult:
    if cls_name in ("DecisionTreeClassifier", "DecisionTreeRegressor"):
        return _ingest_tree(model, sample_range)
    if cls_name == "LogisticRegression":
        return _ingest_logistic(model, sample_range)
    if cls_name in ("MLPClassifier", "MLPRegressor"):
        return _ingest_mlp(model, sample_range)

    raise UnsupportedModelError(
        f"'{path}' unpickles to an unsupported estimator type '{module_name}.{cls_name}'. "
        f"EdgeForge's classical (sklearn) path supports: {', '.join(_SUPPORTED_ESTIMATOR_NAMES)} "
        "-- bare, or wrapped in a Pipeline(StandardScaler|MinMaxScaler, <one of these>)."
    )


def _ingest_pipeline(path: Path, pipeline: Any, sample_range: tuple[float, float] | None) -> IngestResult:
    """Pipeline(StandardScaler|MinMaxScaler, <supported estimator>) -> IngestResult, with the
    scaler folded into a leading "affine" IR node instead of being silently dropped. Any other
    Pipeline shape raises UnsupportedModelError with the same manual-workaround guidance Phase 1
    always gave: extract the final estimator, fold preprocessing into training by hand."""
    steps = list(pipeline.steps)
    if len(steps) != 2:
        raise UnsupportedModelError(
            f"'{path}' unpickles to a sklearn Pipeline with {len(steps)} step(s). EdgeForge only "
            "folds a Pipeline shaped exactly (StandardScaler|MinMaxScaler, <estimator>) -- one "
            "scaler followed by the final estimator -- into the generated C. Pickle the final "
            "step instead, e.g. pickle.dump(pipeline.named_steps['clf'], ...), and fold any other "
            "preprocessing into training."
        )

    (_, scaler), (_, estimator) = steps
    scaler_cls, scaler_module = type(scaler).__name__, type(scaler).__module__
    if not (scaler_module.startswith("sklearn.preprocessing") and scaler_cls in _SUPPORTED_SCALER_NAMES):
        raise UnsupportedModelError(
            f"'{path}' unpickles to a sklearn Pipeline whose first step is "
            f"'{scaler_module}.{scaler_cls}', not a supported scaler. EdgeForge folds "
            f"{' or '.join(_SUPPORTED_SCALER_NAMES)} into the generated C; other preprocessing "
            "(PCA, feature selection, a custom transformer, ...) isn't supported -- fold it into "
            "training and pickle a Pipeline of just a supported scaler + estimator (or the bare "
            "fitted estimator) instead."
        )

    est_cls_name, est_module_name = type(estimator).__name__, type(estimator).__module__
    if est_cls_name not in _SUPPORTED_ESTIMATOR_NAMES:
        raise UnsupportedModelError(
            f"'{path}' unpickles to a sklearn Pipeline whose final step is an unsupported "
            f"estimator type '{est_module_name}.{est_cls_name}'. EdgeForge's classical (sklearn) "
            f"path supports: {', '.join(_SUPPORTED_ESTIMATOR_NAMES)}."
        )

    n_features = int(scaler.n_features_in_)
    scale, shift = _fuse_scaler(path, scaler)

    result = _ingest_bare_estimator(path, estimator, est_cls_name, est_module_name, sample_range=None)
    ir = result.ir

    scale_name, shift_name, scaled_name = "scaler0_scale", "scaler0_shift", "scaler0_out"
    ir.tensors[scale_name] = TensorSpec(scale_name, (n_features,), "float32", role="weight", data=scale)
    ir.tensors[shift_name] = TensorSpec(shift_name, (n_features,), "float32", role="weight", data=shift)
    ir.tensors[scaled_name] = TensorSpec(scaled_name, (n_features,), "float32", role="activation")
    ir.nodes[0].inputs[0] = scaled_name  # the estimator's first node read "input" directly; now reads the scaled buffer
    ir.nodes.insert(0, Node(
        op="affine", name="scaler0",
        inputs=["input"], outputs=[scaled_name],
        attrs={"scale": scale_name, "shift": shift_name},
    ))

    def reference_fn(sample: np.ndarray):
        x = sample.reshape(1, -1)
        if ir.task == "classification":
            proba = pipeline.predict_proba(x)[0]
            return int(np.argmax(proba)), proba.astype(np.float32)
        pred = np.atleast_1d(pipeline.predict(x)[0])
        return None, pred.astype(np.float32)

    return IngestResult(
        ir=ir,
        reference_fn=reference_fn,
        input_sampler=_pipeline_sampler(scaler, n_features, sample_range),
        source_kind="sklearn",
    )


def _fuse_scaler(path: Path, scaler: Any) -> tuple[np.ndarray, np.ndarray]:
    """Returns (scale, shift) such that scaler.transform(x) == x*scale + shift elementwise,
    fusing sklearn's own fitted parameters into the single elementwise affine step the "affine"
    IR op expects."""
    if type(scaler).__name__ == "StandardScaler":
        if scaler.mean_ is None or scaler.scale_ is None:
            raise UnsupportedModelError(
                f"'{path}': the Pipeline's StandardScaler was fit with with_mean=False or "
                "with_std=False, so it isn't a full per-feature affine transform. EdgeForge "
                "needs both centering and scaling enabled (StandardScaler's defaults)."
            )
        std = np.asarray(scaler.scale_, dtype=np.float64)
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        return 1.0 / std, -mean / std
    # MinMaxScaler already fits as X*scale_ + min_ -- see its own source/docstring.
    return np.asarray(scaler.scale_, dtype=np.float64), np.asarray(scaler.min_, dtype=np.float64)


def _pipeline_sampler(scaler: Any, n_features: int, sample_range: tuple[float, float] | None):
    """Golden-vector sampling now needs to cover the *raw* sensor domain (whatever a real
    board's read_sensor() produces), not the standardized/normalized domain
    _DEFAULT_SAMPLE_RANGE describes -- so an explicit --sample-range is honored as-is (it now
    describes the raw domain), and the default is instead derived per-feature from the fitted
    scaler's own statistics, which is the best available guess at the real training-data range."""
    if sample_range is not None:
        lo, hi = sample_range
        return uniform_range_sampler(np.full(n_features, lo), np.full(n_features, hi))
    if type(scaler).__name__ == "StandardScaler":
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        std = np.asarray(scaler.scale_, dtype=np.float64)
        return uniform_range_sampler(mean - 3.0 * std, mean + 3.0 * std)
    return uniform_range_sampler(
        np.asarray(scaler.data_min_, dtype=np.float64), np.asarray(scaler.data_max_, dtype=np.float64)
    )


def _default_sampler(n_features: int, sample_range: tuple[float, float] | None):
    lo, hi = sample_range or _DEFAULT_SAMPLE_RANGE
    return uniform_range_sampler(np.full(n_features, lo), np.full(n_features, hi))


def _tree_derived_sampler(model, sample_range: tuple[float, float] | None):
    """Decision trees carry a strong hint of their meaningful input range in
    tree_.threshold (the split points actually learned from training data).
    Use it as the default sampling range instead of a generic guess, unless the
    caller overrode it explicitly."""
    if sample_range is not None:
        return _default_sampler(model.n_features_in_, sample_range)
    t = model.tree_
    n_features = model.n_features_in_
    lo = np.full(n_features, _DEFAULT_SAMPLE_RANGE[0])
    hi = np.full(n_features, _DEFAULT_SAMPLE_RANGE[1])
    is_split = t.feature >= 0
    if np.any(is_split):
        for f in range(n_features):
            mask = is_split & (t.feature == f)
            if np.any(mask):
                vals = t.threshold[mask]
                span = float(vals.max() - vals.min()) or 1.0
                pad = span * 0.3 + 1e-6
                lo[f] = float(vals.min() - pad)
                hi[f] = float(vals.max() + pad)
    return uniform_range_sampler(lo, hi)


def _ingest_tree(model, sample_range: tuple[float, float] | None) -> IngestResult:
    is_classifier = type(model).__name__ == "DecisionTreeClassifier"
    t = model.tree_
    n_features = model.n_features_in_
    n_nodes = t.node_count

    feature = t.feature.astype(np.int32).tolist()  # -2 (TREE_UNDEFINED) at leaves, matches sklearn's own sentinel
    threshold = t.threshold.astype(np.float64).tolist()
    left = t.children_left.astype(np.int32).tolist()
    right = t.children_right.astype(np.int32).tolist()

    if is_classifier:
        n_classes = len(model.classes_)
        leaf_value = [float(np.argmax(t.value[i, 0, :])) for i in range(n_nodes)]
        out_size = n_classes
        task = "classification"
    else:
        leaf_value = [float(t.value[i, 0, 0]) for i in range(n_nodes)]
        out_size = 1
        task = "regression"

    # Tree data is real constant (flash-resident) storage, so it's represented as ordinary
    # weight TensorSpecs -- same as every other op -- rather than raw attrs, so footprint
    # accounting and const-array codegen both work generically across every op kind.
    input_spec = TensorSpec("input", (n_features,), "float32", role="input")
    output_spec = TensorSpec("output", (out_size,), "float32", role="output")
    feature_spec = TensorSpec("tree0_feature", (n_nodes,), "int32", role="weight", data=np.array(feature, dtype=np.int32))
    threshold_spec = TensorSpec("tree0_threshold", (n_nodes,), "float32", role="weight", data=np.array(threshold, dtype=np.float64))
    left_spec = TensorSpec("tree0_left", (n_nodes,), "int32", role="weight", data=np.array(left, dtype=np.int32))
    right_spec = TensorSpec("tree0_right", (n_nodes,), "int32", role="weight", data=np.array(right, dtype=np.int32))
    leaf_value_spec = TensorSpec("tree0_leaf_value", (n_nodes,), "float32", role="weight", data=np.array(leaf_value, dtype=np.float64))
    node = Node(
        op="tree",
        name="tree0",
        inputs=["input"],
        outputs=["output"],
        attrs={
            "n_nodes": n_nodes,
            "feature": "tree0_feature",
            "threshold": "tree0_threshold",
            "left": "tree0_left",
            "right": "tree0_right",
            "leaf_value": "tree0_leaf_value",
            "is_classifier": is_classifier,
            "out_size": out_size,
        },
    )
    ir = ModelIR(
        kind="classical",
        task=task,
        input_spec=input_spec,
        output_spec=output_spec,
        tensors={
            "input": input_spec,
            "output": output_spec,
            "tree0_feature": feature_spec,
            "tree0_threshold": threshold_spec,
            "tree0_left": left_spec,
            "tree0_right": right_spec,
            "tree0_leaf_value": leaf_value_spec,
        },
        nodes=[node],
        metadata={
            "source_framework": "sklearn",
            "model_class": type(model).__name__,
            "class_labels": [str(c) for c in model.classes_] if is_classifier else None,
        },
    )

    def reference_fn(sample: np.ndarray):
        x = sample.reshape(1, -1)
        if is_classifier:
            proba = model.predict_proba(x)[0]
            return int(np.argmax(proba)), proba.astype(np.float32)
        pred = model.predict(x)[0]
        return None, np.array([pred], dtype=np.float32)

    return IngestResult(
        ir=ir,
        reference_fn=reference_fn,
        input_sampler=_tree_derived_sampler(model, sample_range),
        source_kind="sklearn",
    )


def _binary_expand(coef: np.ndarray, intercept: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sklearn stores binary LogisticRegression as a single sigmoid row. Expand
    to two rows (class 0 fixed at logit 0, class 1 = the learned row) so argmax
    over the two logits reproduces sigmoid(z) >= 0.5 exactly, keeping the
    'classification = argmax over out_size logits' contract uniform."""
    n_features = coef.shape[1]
    row0 = np.zeros((1, n_features), dtype=coef.dtype)
    b0 = np.zeros((1,), dtype=intercept.dtype)
    return np.concatenate([row0, coef], axis=0), np.concatenate([b0, intercept], axis=0)


def _ingest_logistic(model, sample_range: tuple[float, float] | None) -> IngestResult:
    n_features = model.n_features_in_
    n_classes = len(model.classes_)
    coef = model.coef_.astype(np.float64)
    intercept = model.intercept_.astype(np.float64)
    if coef.shape[0] == 1 and n_classes == 2:
        coef, intercept = _binary_expand(coef, intercept)
    out_size = coef.shape[0]

    input_spec = TensorSpec("input", (n_features,), "float32", role="input")
    output_spec = TensorSpec("output", (out_size,), "float32", role="output")
    w = TensorSpec("w0", (out_size, n_features), "float32", role="weight", data=coef)
    b = TensorSpec("b0", (out_size,), "float32", role="bias", data=intercept)
    node = Node(
        op="linear",
        name="fc0",
        inputs=["input"],
        outputs=["output"],
        attrs={"weight": "w0", "bias": "b0", "n_in": n_features, "n_out": out_size},
    )
    ir = ModelIR(
        kind="classical",
        task="classification",
        input_spec=input_spec,
        output_spec=output_spec,
        tensors={"input": input_spec, "output": output_spec, "w0": w, "b0": b},
        nodes=[node],
        metadata={
            "source_framework": "sklearn",
            "model_class": "LogisticRegression",
            "class_labels": [str(c) for c in model.classes_],
            "output_is_raw_logit": True,
        },
    )

    def reference_fn(sample: np.ndarray):
        x = sample.reshape(1, -1)
        proba = model.predict_proba(x)[0]
        return int(np.argmax(proba)), proba.astype(np.float32)

    return IngestResult(
        ir=ir,
        reference_fn=reference_fn,
        input_sampler=_default_sampler(n_features, sample_range),
        source_kind="sklearn",
    )


def _ingest_mlp(model, sample_range: tuple[float, float] | None) -> IngestResult:
    is_classifier = type(model).__name__ == "MLPClassifier"
    n_features = model.n_features_in_
    hidden_activation = model.activation
    if hidden_activation not in _SAFE_HIDDEN_ACTIVATIONS:
        raise UnsupportedOpError(
            f"MLP hidden activation '{hidden_activation}' is not supported: it requires a "
            "transcendental function (exp/tanh), which EdgeForge's freestanding codegen doesn't "
            "link on memory-constrained targets. Retrain with activation='relu' (sklearn's default)."
        )

    n_layers = len(model.coefs_)
    task = "classification" if is_classifier else "regression"
    input_spec = TensorSpec("input", (n_features,), "float32", role="input")

    # Classification always elides the final softmax/logistic (argmax is invariant to it).
    # Regression keeps it only if it's not a no-op identity, and only if it's dependency-free.
    if is_classifier:
        emit_final_activation = False
    else:
        out_act = model.out_activation_
        if out_act == "identity":
            emit_final_activation = False
        elif out_act in _SAFE_HIDDEN_ACTIVATIONS:
            emit_final_activation = True
        else:
            raise UnsupportedOpError(
                f"MLPRegressor output activation '{out_act}' is not supported (needs a transcendental "
                "function). Use out_activation_='identity' (sklearn's default for regression)."
            )

    tensors: dict[str, TensorSpec] = {"input": input_spec}
    nodes: list[Node] = []
    prev_name = "input"

    for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        w_np = coef.astype(np.float64).T  # sklearn stores (fan_in, fan_out); IR wants (n_out, n_in)
        b_np = intercept.astype(np.float64)
        n_out, n_in = w_np.shape
        is_last = i == n_layers - 1

        if is_last:
            buf_name, role = ("output", "output") if not emit_final_activation else ("final_preact", "activation")
        else:
            buf_name, role = f"hidden{i}", "activation"

        tensors[f"w{i}"] = TensorSpec(f"w{i}", (n_out, n_in), "float32", role="weight", data=w_np)
        tensors[f"b{i}"] = TensorSpec(f"b{i}", (n_out,), "float32", role="bias", data=b_np)
        tensors[buf_name] = TensorSpec(buf_name, (n_out,), "float32", role=role)
        nodes.append(
            Node("linear", f"fc{i}", [prev_name], [buf_name], {"weight": f"w{i}", "bias": f"b{i}", "n_in": n_in, "n_out": n_out})
        )

        if not is_last:
            # In-place: relu(hidden_i) overwrites its own buffer, so a hidden layer costs one
            # scratch buffer total instead of a pre- and a post-activation copy.
            nodes.append(Node("activation", f"act{i}", [buf_name], [buf_name], {"kind": hidden_activation}))

        prev_name = buf_name

    if emit_final_activation:
        n_out = tensors["final_preact"].shape[0]
        tensors["output"] = TensorSpec("output", (n_out,), "float32", role="output")
        nodes.append(Node("activation", "act_out", ["final_preact"], ["output"], {"kind": model.out_activation_}))

    output_spec = tensors["output"]

    ir = ModelIR(
        kind="classical",
        task=task,
        input_spec=input_spec,
        output_spec=output_spec,
        tensors=tensors,
        nodes=nodes,
        metadata={
            "source_framework": "sklearn",
            "model_class": type(model).__name__,
            "hidden_activation": hidden_activation,
            "n_layers": n_layers,
            "class_labels": [str(c) for c in model.classes_] if is_classifier else None,
        },
    )

    def reference_fn(sample: np.ndarray):
        x = sample.reshape(1, -1)
        if is_classifier:
            proba = model.predict_proba(x)[0]
            return int(np.argmax(proba)), proba.astype(np.float32)
        pred = np.atleast_1d(model.predict(x)[0])
        return None, pred.astype(np.float32)

    return IngestResult(
        ir=ir,
        reference_fn=reference_fn,
        input_sampler=_default_sampler(n_features, sample_range),
        source_kind="sklearn",
    )
