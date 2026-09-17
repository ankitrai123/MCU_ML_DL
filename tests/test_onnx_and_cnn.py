"""Regression coverage for the two trickiest codegen paths: ONNX Gemm/MatMul+Add
feedforward ingestion, and the deep-tier conv2d/depthwise_conv2d/maxpool2d chain
(including the elided-Flatten aliasing that caused a real shape/indexing bug during
development -- see flat_ref in codegen/formatting.py)."""

from edgeforge.boards.registry import BoardRegistry
from edgeforge.codegen.generator import generate
from edgeforge.validate.golden import run_golden_validation


def test_onnx_mlp_ingest_and_validate(onnx_mlp_path, boards_dir, tmp_path):
    from edgeforge.ingest import onnx_ingest

    result = onnx_ingest.ingest(onnx_mlp_path)
    assert result.ir.kind == "classical"
    assert result.ir.task == "classification"

    reg = BoardRegistry(boards_dir)
    board = reg.get("stm32f411")
    gen = generate(result.ir, board, tmp_path)
    report = run_golden_validation(result, board, gen.model_c, tmp_path, n_samples=20)
    assert report.all_passed, report.describe()


def test_cnn_ingest_elides_flatten_and_validates(keras_cnn_path, boards_dir, tmp_path):
    from edgeforge.ingest import keras_ingest

    result = keras_ingest.ingest(keras_cnn_path, sample_range=(0, 1))
    ir = result.ir
    ops = [n.op for n in ir.nodes]
    assert "conv2d" in ops and "depthwise_conv2d" in ops and "maxpool2d" in ops
    # No explicit reshape/flatten op should ever reach the IR -- see keras_ingest's docstring.
    assert "reshape" not in ops and "flatten" not in ops

    reg = BoardRegistry(boards_dir)
    board = reg.get("stm32f411")
    gen = generate(ir, board, tmp_path)
    report = run_golden_validation(result, board, gen.model_c, tmp_path, n_samples=25)
    assert report.all_passed, report.describe()
