"""Verilog/Lattice backend (Phase 3, second FPGA backend): every op in edgeforge.ir.SUPPORTED_OPS
compiled to synthesizable RTL by a from-scratch generator -- there's no existing "hls4ml for the
open-source Lattice flow" tool to wrap (see edgeforge.verilog.convert's module docstring), so
EdgeForge generates the RTL itself here, host verified via Icarus Verilog simulation against the
real source model (sklearn's own predictions for the classical/Q16.16 tier, the real TFLite
Interpreter for the deep/int8 tier). A deliberately separate path from `convert`/`--board`: no
boards/*.yaml, no edgeforge.codegen involvement, its own CLI subcommand.
"""

import shutil

import pytest

from edgeforge.cli import main
from edgeforge.errors import ToolchainNotFoundError, UnsupportedOpError
from edgeforge.ingest import keras_ingest, sklearn_ingest
from edgeforge.verilog.codegen import validate_codegen_support
from edgeforge.verilog.convert import convert_and_validate
from edgeforge.verilog.simulate import run_verilog_simulation

requires_iverilog = pytest.mark.skipif(
    shutil.which("iverilog") is None or shutil.which("vvp") is None, reason="iverilog/vvp not installed"
)
requires_ecp5_toolchain = pytest.mark.skipif(
    any(shutil.which(t) is None for t in ("yosys", "nextpnr-ecp5", "ecppack")),
    reason="yosys/nextpnr-ecp5/ecppack not installed",
)


@pytest.fixture(scope="session")
def keras_deep_regression_path(tmp_path_factory):
    """A deep-tier (int8/TFLite) *regression* model -- the one deep-tier task shape
    test_deep_mlp_matches_tflite_interpreter/test_deep_cnn_matches_tflite_interpreter don't
    cover (both are classifiers), needed to verify simulate.py's int8 output dequantization."""
    tf = pytest.importorskip("tensorflow")
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(200, 4)).astype(np.float32)
    y = (X[:, 0] * 2 - X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)

    tf.random.set_seed(0)
    inputs = tf.keras.Input(shape=(4,))
    h = tf.keras.layers.Dense(6, activation="relu")(inputs)
    out = tf.keras.layers.Dense(1, activation="linear")(h)
    model = tf.keras.Model(inputs, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=30, verbose=0)

    path = tmp_path_factory.mktemp("models") / "deep_regr.keras"
    model.save(path)
    return path


@requires_iverilog
def test_tree_classifier_matches_sklearn(tree_clf_path, tmp_path):
    result = sklearn_ingest.ingest(tree_clf_path)
    report = run_verilog_simulation(result, tmp_path, n_samples=20, seed=0)
    assert report.all_passed, report.describe()


@requires_iverilog
def test_linear_classifier_matches_sklearn(logreg_bin_path, tmp_path):
    result = sklearn_ingest.ingest(logreg_bin_path)
    report = run_verilog_simulation(result, tmp_path, n_samples=20, seed=0)
    assert report.all_passed, report.describe()


@requires_iverilog
def test_scaler_pipeline_matches_sklearn(scaled_logreg_path, tmp_path):
    """Exercises affine -> linear chaining (a folded StandardScaler ahead of the estimator)
    and an intermediate activation-tensor buffer shared between two nodes."""
    result = sklearn_ingest.ingest(scaled_logreg_path)
    assert result.ir.nodes[0].op == "affine"
    report = run_verilog_simulation(result, tmp_path, n_samples=20, seed=0)
    assert report.all_passed, report.describe()


@requires_iverilog
def test_mlp_matches_sklearn(mlp_clf_path, tmp_path):
    """Exercises linear -> activation(relu, in place) -> linear chaining."""
    result = sklearn_ingest.ingest(mlp_clf_path)
    assert any(n.op == "activation" for n in result.ir.nodes)
    report = run_verilog_simulation(result, tmp_path, n_samples=20, seed=0)
    assert report.all_passed, report.describe()


@requires_iverilog
def test_deep_mlp_matches_tflite_interpreter(keras_mlp_path, tmp_path):
    """Exercises the deep (int8) tier's own linear op and integer requantization
    (edgeforge/verilog/quant_math.py) against the real TFLite Interpreter, not the original
    float Keras model -- the same reference every other deep-tier check in this project uses."""
    result = keras_ingest.ingest(keras_mlp_path, sample_range=(-2, 8))
    assert result.ir.tensors[result.ir.nodes[0].attrs["weight"]].dtype == "int8"
    report = run_verilog_simulation(result, tmp_path, n_samples=20, seed=0)
    assert report.all_passed, report.describe()


@requires_iverilog
def test_deep_cnn_matches_tflite_interpreter(keras_cnn_path, tmp_path):
    """Exercises conv2d, maxpool2d, and depthwise_conv2d together -- found and fixed two real
    bugs building this: an 8-bit slice-assignment that dropped sign extension on every negative
    int8 value, and depthwise_conv2d wrongly reusing plain conv2d's per-input-channel loop
    (depthwise has none -- each output channel reads exactly one fixed input channel)."""
    result = keras_ingest.ingest(keras_cnn_path, sample_range=(0.0, 1.0))
    ops = [n.op for n in result.ir.nodes]
    assert ops == ["conv2d", "maxpool2d", "depthwise_conv2d", "linear"]
    report = run_verilog_simulation(result, tmp_path, n_samples=20, seed=0)
    assert report.all_passed, report.describe()


@requires_iverilog
def test_deep_regression_output_dequantized_correctly(keras_deep_regression_path, tmp_path):
    """The one deep-tier task shape the other deep-tier tests don't cover: a regression output
    must be dequantized via its own tensor's TFLite scale/zero_point (simulate.py's
    _dequantize_output), not the classical tier's Q16.16 -- classification tests can't catch
    this bug since they only ever check the predicted class, never the raw output value."""
    result = keras_ingest.ingest(keras_deep_regression_path, sample_range=(0.0, 1.0), task="regression")
    assert result.ir.kind == "deep"
    assert result.ir.task == "regression"
    report = run_verilog_simulation(result, tmp_path, n_samples=15, seed=0)
    assert report.all_passed, report.describe()
    for s in report.samples:
        assert abs(s.got_output[0] - s.ref_output[0]) < 1e-3


def test_transcendental_activation_rejected_with_a_clear_error():
    """No EdgeForge backend emits a transcendental function (softmax/sigmoid/tanh) in hardware
    -- reaching codegen with one is an ingest bug, not a model problem, and must fail loudly."""
    from edgeforge.ir import ModelIR, Node, TensorSpec

    x = TensorSpec("x", (2,), "float32", role="input")
    ir = ModelIR(
        kind="classical",
        task="regression",
        input_spec=x,
        output_spec=x,  # the in-place activation's output IS the model's only tensor
        tensors={"x": x},
        nodes=[Node("activation", "act0", ["x"], ["x"], attrs={"kind": "sigmoid"})],
    )
    with pytest.raises(UnsupportedOpError, match="internal error"):
        validate_codegen_support(ir)


@requires_iverilog
def test_missing_iverilog_reports_actionable_error(tree_clf_path, tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    result = sklearn_ingest.ingest(tree_clf_path)
    with pytest.raises(ToolchainNotFoundError, match="iverilog"):
        run_verilog_simulation(result, tmp_path)


@requires_iverilog
def test_synthesize_without_ecp5_toolchain_degrades_gracefully(tree_clf_path, tmp_path, monkeypatch):
    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda name: None if name in ("yosys", "nextpnr-ecp5", "ecppack") else real_which(name)
    )
    result = convert_and_validate(tree_clf_path, tmp_path, n_samples=10, attempt_synthesis=True)
    assert result.ok  # simulation is unaffected by the ECP5 toolchain being unavailable
    assert not result.synth_attempted
    assert result.toolchain_missing in ("yosys", "nextpnr-ecp5", "ecppack")
    assert result.install_hint


@requires_ecp5_toolchain
def test_synthesize_with_ecp5_toolchain(tree_clf_path, tmp_path):
    result = convert_and_validate(tree_clf_path, tmp_path, n_samples=5, attempt_synthesis=True)
    assert result.synth_attempted
    assert result.synth_ok, result.error
    assert "Device utilisation" in result.resource_report


@requires_iverilog
def test_cli_convert_verilog_succeeds(tree_clf_path, tmp_path, capsys):
    rc = main(["convert-verilog", "--model", str(tree_clf_path), "--out", str(tmp_path), "--samples", "10"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "convert-verilog: SUCCESS" in out
    assert "Icarus Verilog simulation" in out


@requires_iverilog
def test_cli_convert_verilog_succeeds_for_deep_tier(keras_mlp_path, tmp_path, capsys):
    rc = main([
        "convert-verilog", "--model", str(keras_mlp_path), "--out", str(tmp_path),
        "--samples", "10",
    ])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "convert-verilog: SUCCESS" in out


@requires_ecp5_toolchain
def test_synthesize_reports_io_pin_exhaustion_cleanly(keras_cnn_path, tmp_path):
    """A model with enough input+output elements needs more top-level I/O pins than a real
    ECP5 package has (every port here is a raw flattened bus -- see convert.py's module
    docstring) -- place-and-route genuinely can't succeed, and that must come back as a clear,
    handled failure (synth_ok=False + a real error), never a crash, and never mistaken for a
    correctness problem (simulation, which doesn't touch pins, is unaffected)."""
    build_result = convert_and_validate(keras_cnn_path, tmp_path, n_samples=3, attempt_synthesis=True)
    assert build_result.ok  # correctness is independent of whether it fits on this device
    assert build_result.synth_attempted
    assert not build_result.synth_ok
    assert "place-and-route failed" in build_result.error
