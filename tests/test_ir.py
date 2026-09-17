import numpy as np

from edgeforge.ir import ModelIR, Node, TensorSpec, compute_asymmetric_quant_params, dequantize_affine, quantize_affine


def test_tensor_spec_size_and_bytes():
    t = TensorSpec("w", (3, 4), "float32", role="weight", data=np.zeros((3, 4)))
    assert t.size == 12
    assert t.nbytes == 48


def test_model_ir_weight_and_activation_bytes_are_disjoint():
    x = TensorSpec("input", (4,), "float32", role="input")
    y = TensorSpec("output", (3,), "float32", role="output")
    hidden = TensorSpec("hidden0", (5,), "float32", role="activation")
    w = TensorSpec("w0", (5, 4), "float32", role="weight", data=np.zeros((5, 4)))
    b = TensorSpec("b0", (5,), "float32", role="bias", data=np.zeros(5))
    ir = ModelIR(
        kind="classical", task="classification", input_spec=x, output_spec=y,
        tensors={"input": x, "output": y, "hidden0": hidden, "w0": w, "b0": b},
        nodes=[Node("linear", "fc0", ["input"], ["hidden0"], {"weight": "w0", "bias": "b0"})],
    )
    assert ir.weight_bytes() == (5 * 4 + 5) * 4
    assert ir.activation_bytes() == 5 * 4  # only "hidden0"; input/output excluded by name
    assert ir.io_bytes() == (4 + 3) * 4


def test_quantize_dequantize_roundtrip():
    scale, zp = compute_asymmetric_quant_params(-2.0, 3.0)
    for real in (-2.0, -1.0, 0.0, 1.5, 3.0):
        q = quantize_affine(real, scale, zp)
        assert -128 <= q <= 127
        back = dequantize_affine(q, scale, zp)
        assert abs(back - real) <= scale  # within one quantization step


def test_quantize_affine_clamps_out_of_range():
    scale, zp = compute_asymmetric_quant_params(0.0, 1.0)
    assert quantize_affine(1000.0, scale, zp) == 127
    assert quantize_affine(-1000.0, scale, zp) == -128
