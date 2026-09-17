import pytest

from edgeforge.boards.registry import BoardRegistry
from edgeforge.errors import FootprintExceededError, TierMismatchError
from edgeforge.ingest import sklearn_ingest
from edgeforge.quantize.footprint import check_budget, estimate_footprint


def test_small_tree_fits_every_board(tree_clf_path, boards_dir):
    reg = BoardRegistry(boards_dir)
    result = sklearn_ingest.ingest(tree_clf_path)
    for board_id in ("stm32f411", "8051_at89s52", "native_cortex_m4"):
        report = check_budget(result.ir, reg.get(board_id))
        assert report.fits


def test_deep_model_rejected_by_classical_only_board(keras_mlp_path, boards_dir):
    from edgeforge.ingest import keras_ingest

    reg = BoardRegistry(boards_dir)
    result = keras_ingest.ingest(keras_mlp_path, sample_range=(-2, 8))
    with pytest.raises(TierMismatchError):
        check_budget(result.ir, reg.get("8051_at89s52"))


def test_oversized_model_rejected_before_codegen(tree_clf_path, boards_dir):
    reg = BoardRegistry(boards_dir)
    board = reg.get("8051_at89s52")
    result = sklearn_ingest.ingest(tree_clf_path)
    report = estimate_footprint(result.ir, board)
    assert report.estimated_flash_bytes < board.memory.available_flash_bytes

    # A board with almost no flash at all should reject even this small tree.
    import dataclasses

    tiny_board = dataclasses.replace(board, memory=dataclasses.replace(board.memory, flash_bytes=10, flash_reserved_bytes=0))
    with pytest.raises(FootprintExceededError):
        check_budget(result.ir, tiny_board)
