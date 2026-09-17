"""Numeric derivations codegen templates need but Jinja2 shouldn't compute inline.

EdgeForge's deep-tier boards (Cortex-M / Xtensa) all have usable float
arithmetic (hardware FPU or a soft-float libgcc/newlib), so the int8
requantization step here uses a plain float multiplier rather than TFLite
Micro's integer-only fixed-point-multiply-and-shift trick -- that trick
exists specifically for platforms *without* float, which none of EdgeForge's
boards are (8051 never reaches this code path at all: it's classical-tier
only). A float multiplier is simpler to generate and verify correctly while
landing within the same few-ULP tolerance a fixed-point implementation would.
"""

from __future__ import annotations

from edgeforge.ir import ModelIR, Node, quantize_affine


def requant_multipliers(node: Node, ir: ModelIR) -> list[float]:
    """Per-output-channel (input_scale * weight_scale[c]) / output_scale, TFLite's
    standard int8 requantization multiplier. Channel count is read from the *output*
    tensor's last dimension, which is correct for linear (1-D) and conv2d/depthwise_conv2d
    (H,W,C) alike, and for depthwise's differently-laid-out weight tensor."""
    in_t = ir.tensors[node.inputs[0]]
    w_t = ir.tensors[node.attrs["weight"]]
    out_t = ir.tensors[node.outputs[0]]
    in_scale = in_t.scale[0]
    out_scale = out_t.scale[0]
    n_out = out_t.shape[-1]
    w_scales = w_t.scale if len(w_t.scale) == n_out else tuple(w_t.scale[0] for _ in range(n_out))
    return [(in_scale * ws) / out_scale for ws in w_scales]


def fused_activation_bounds(node: Node, ir: ModelIR) -> tuple[int, int]:
    """Quantized [min, max] clamp for a node's fused activation, in the *output*
    tensor's own quantized domain (relu's real-valued 0 floor becomes
    output_zero_point once quantized, not literal 0)."""
    out_t = ir.tensors[node.outputs[0]]
    scale, zp = out_t.scale[0], out_t.zero_point[0]
    kind = node.attrs.get("fused_activation", "none")
    qmin, qmax = -128, 127
    if kind == "relu":
        qmin = zp
    elif kind == "relu6":
        qmin = zp
        qmax = min(127, quantize_affine(6.0, scale, zp))
    return qmin, qmax
