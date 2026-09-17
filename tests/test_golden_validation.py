import pytest

from edgeforge.boards.registry import BoardRegistry
from edgeforge.codegen.generator import generate
from edgeforge.errors import ValidationError
from edgeforge.ingest import sklearn_ingest
from edgeforge.validate.golden import run_golden_validation


def test_golden_validation_passes_tree_on_stm32(tree_clf_path, boards_dir, tmp_path):
    reg = BoardRegistry(boards_dir)
    result = sklearn_ingest.ingest(tree_clf_path)
    board = reg.get("stm32f411")
    gen = generate(result.ir, board, tmp_path)
    report = run_golden_validation(result, board, gen.model_c, tmp_path, n_samples=15)
    assert report.all_passed, report.describe()


def test_golden_validation_passes_tree_on_8051_profile(tree_clf_path, boards_dir, tmp_path):
    # Golden validation always compiles with the host compiler, never the board's own
    # toolchain, so this exercises the 8051 board's c_dialect (the __code/__SDCC guard
    # rendering as nothing under host gcc) without needing sdcc installed.
    reg = BoardRegistry(boards_dir)
    result = sklearn_ingest.ingest(tree_clf_path)
    board = reg.get("8051_at89s52")
    gen = generate(result.ir, board, tmp_path)
    report = run_golden_validation(result, board, gen.model_c, tmp_path, n_samples=15)
    assert report.all_passed, report.describe()


def test_golden_validation_passes_deep_mlp(keras_mlp_path, boards_dir, tmp_path):
    from edgeforge.ingest import keras_ingest

    reg = BoardRegistry(boards_dir)
    result = keras_ingest.ingest(keras_mlp_path, sample_range=(-2, 8))
    board = reg.get("stm32f411")
    gen = generate(result.ir, board, tmp_path)
    report = run_golden_validation(result, board, gen.model_c, tmp_path, n_samples=20)
    assert report.all_passed, report.describe()


def test_golden_validation_detects_a_real_mismatch(tree_clf_path, boards_dir, tmp_path):
    """A meta-test: corrupt a weight after generation and confirm validation catches it,
    so a passing suite isn't just because the harness never actually checks anything."""
    reg = BoardRegistry(boards_dir)
    result = sklearn_ingest.ingest(tree_clf_path)
    board = reg.get("stm32f411")
    gen = generate(result.ir, board, tmp_path)

    text = gen.model_c.read_text()
    # Flip the first leaf class assignment so at least one prediction path changes.
    corrupted = text.replace("(ef_k == ef_predicted)", "(ef_k != ef_predicted)", 1)
    assert corrupted != text
    gen.model_c.write_text(corrupted)

    report = run_golden_validation(result, board, gen.model_c, tmp_path, n_samples=20)
    assert not report.all_passed


def test_missing_host_compiler_raises_clear_message(tree_clf_path, boards_dir, tmp_path, monkeypatch):
    """A user with no host C compiler at all (e.g. a fresh Windows machine) must get a
    message naming the missing tool and how to install it, not an empty-log failure --
    regression test for exactly that gap."""
    reg = BoardRegistry(boards_dir)
    result = sklearn_ingest.ingest(tree_clf_path)
    board = reg.get("stm32f411")
    gen = generate(result.ir, board, tmp_path)

    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ValidationError, match="host C compiler"):
        run_golden_validation(result, board, gen.model_c, tmp_path, n_samples=5)
