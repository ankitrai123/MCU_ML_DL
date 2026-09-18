"""Verilog/Lattice backend (Phase 3, second FPGA backend): classical-tier (tree/linear/
affine/activation) models compiled to synthesizable RTL by a from-scratch generator -- there's
no existing "hls4ml for the open-source Lattice flow" tool to wrap (see
edgeforge.verilog.convert's module docstring), so EdgeForge generates the RTL itself here, host
verified via Icarus Verilog simulation against the real sklearn model. A deliberately separate
path from `convert`/`--board`: no boards/*.yaml, no edgeforge.ir/edgeforge.codegen involvement,
its own CLI subcommand.
"""

import shutil

import pytest

from edgeforge.cli import main
from edgeforge.errors import ToolchainNotFoundError, UnsupportedOpError
from edgeforge.ingest import sklearn_ingest
from edgeforge.verilog.codegen import validate_classical_tier
from edgeforge.verilog.convert import convert_and_validate
from edgeforge.verilog.simulate import run_verilog_simulation

requires_iverilog = pytest.mark.skipif(
    shutil.which("iverilog") is None or shutil.which("vvp") is None, reason="iverilog/vvp not installed"
)
requires_ecp5_toolchain = pytest.mark.skipif(
    any(shutil.which(t) is None for t in ("yosys", "nextpnr-ecp5", "ecppack")),
    reason="yosys/nextpnr-ecp5/ecppack not installed",
)


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


def test_deep_tier_node_rejected_with_a_clear_error(keras_mlp_path):
    """conv2d/depthwise_conv2d/maxpool2d/int8-quantized linear aren't implemented in this
    increment -- ingest must fail loudly and clearly, never silently emit wrong RTL."""
    from edgeforge.ingest import keras_ingest

    result = keras_ingest.ingest(keras_mlp_path, sample_range=(-2, 8))
    with pytest.raises(UnsupportedOpError, match="classical tier"):
        validate_classical_tier(result.ir)


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


def test_cli_convert_verilog_rejects_deep_tier_cleanly(keras_mlp_path, tmp_path, capsys):
    rc = main(["convert-verilog", "--model", str(keras_mlp_path), "--out", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "classical tier" in captured.err
