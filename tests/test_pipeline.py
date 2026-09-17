"""Tests for edgeforge.pipeline.run_conversion, the orchestration both the CLI
and the web UI call -- in particular that a machine with no compilers installed
at all (a first-time non-technical user's likely starting point) still gets a
successful result with generated C, not a hard failure."""

from pathlib import Path

from edgeforge.boards.registry import BoardRegistry
from edgeforge.pipeline import run_conversion


def test_succeeds_end_to_end_with_all_toolchains_present(tree_clf_path, boards_dir, tmp_path):
    reg = BoardRegistry(boards_dir)
    result = run_conversion(tree_clf_path, reg.get("stm32f411"), tmp_path)
    assert result.ok
    assert result.error is None
    assert result.golden is not None
    assert result.golden.all_passed


def test_missing_every_toolchain_still_succeeds_with_generated_source(tree_clf_path, boards_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    reg = BoardRegistry(boards_dir)
    result = run_conversion(tree_clf_path, reg.get("stm32f411"), tmp_path)

    assert result.ok  # the headline outcome: this must not read as a failure
    assert result.error is None
    assert result.generated is not None
    assert result.generated.model_c.is_file()
    assert result.build is not None and result.build.toolchain_missing == "arm-none-eabi-gcc"
    assert result.golden is None
    assert result.validate_skipped_reason is not None
    assert "host C compiler" in result.validate_skipped_reason


def test_missing_toolchains_does_not_mask_a_real_footprint_failure(keras_mlp_path, boards_dir, tmp_path, monkeypatch):
    """Missing tools degrade gracefully, but a model that genuinely doesn't fit the
    board must still be reported as an error, not silently swallowed alongside them."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    reg = BoardRegistry(boards_dir)
    result = run_conversion(keras_mlp_path, reg.get("8051_at89s52"), tmp_path, sample_range=(-2, 8))
    assert not result.ok
    assert result.error_stage == "footprint"
