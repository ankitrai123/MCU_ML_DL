"""Renders synthesizable Verilog + a host-simulation testbench from a ModelIR.

Every op in edgeforge.ir.SUPPORTED_OPS is implemented: tree/linear(float32)/affine/activation
in Q16.16 fixed point for the classical (sklearn) tier, and linear(int8)/conv2d/
depthwise_conv2d/maxpool2d in plain int8/int32 with an integer multiply-shift requantization
(edgeforge/verilog/quant_math.py) for the deep (TFLite) tier -- the same op set the C codegen
supports, reusing the exact same ingest backends and IR, just emitting a different (HDL, not C)
target language. An activation node needing a transcendental function (softmax/sigmoid/tanh)
still can't be emitted in hardware by any EdgeForge backend and raises a clear error, exactly
like the C codegen's own equivalent check.

A deep-tier *regression* model's raw output is dequantized via its own tensor's TFLite
scale/zero_point (edgeforge/verilog/simulate.py's `_dequantize_output`), the same way a
classical-tier one uses Q16.16 -- confirmed against a real Dense(relu)->Dense(linear) TFLite
regression model, matching the TFLite Interpreter's own output to ~1e-6 across every sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from edgeforge import __version__
from edgeforge.codegen.quant_math import fused_activation_bounds
from edgeforge.errors import UnsupportedOpError
from edgeforge.ir import ModelIR, quantize_affine
from edgeforge.verilog import formatting
from edgeforge.verilog.fixedpoint import fixed_literal, to_fixed_array, verilog_signed_literal
from edgeforge.verilog.quant_math import requant_multipliers_for_node

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_TRANSCENDENTAL_ACTIVATIONS = ("sigmoid", "tanh", "logistic", "softmax")


def make_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.globals.update(
        buf_name=formatting.buf_name,
        tensor_init_lines=formatting.tensor_init_lines,
        fixed_literal=fixed_literal,
        verilog_signed_literal=verilog_signed_literal,
        int_literal=lambda x: str(int(x)),
        requant_multipliers_for_node=requant_multipliers_for_node,
        fused_activation_bounds=fused_activation_bounds,
    )
    return env


def validate_codegen_support(ir: ModelIR) -> None:
    """Raises UnsupportedOpError naming exactly what's missing, rather than letting an
    unhandled node silently fall through Jinja2's StrictUndefined into a confusing template
    error (or, worse, an empty/wrong RTL block). Every op in edgeforge.ir.SUPPORTED_OPS has a
    real implementation here; the only remaining rejection (mirroring the C codegen's own) is
    an activation that needs a transcendental function no EdgeForge backend emits in hardware."""
    for node in ir.nodes:
        if node.op == "activation" and node.attrs.get("kind") in _TRANSCENDENTAL_ACTIVATIONS:
            raise UnsupportedOpError(
                f"internal error: node '{node.name}' (activation kind={node.attrs['kind']!r}) reached "
                "Verilog codegen without being elided or rejected at ingest -- this is an EdgeForge "
                "bug, not a model problem; please report it."
            )
        if node.op == "maxpool2d":
            # The pooling kernel passes raw quantized values straight through (max, or an
            # integer average) with no rescale step -- only valid when input/output share
            # quantization params, exactly the same invariant the C codegen checks (and relies
            # on) for the same reason; see edgeforge/codegen/context.py's own copy of this check.
            in_t = ir.tensors[node.inputs[0]]
            out_t = ir.tensors[node.outputs[0]]
            if in_t.scale != out_t.scale or in_t.zero_point != out_t.zero_point:
                raise UnsupportedOpError(
                    f"internal error: pooling node '{node.name}' has mismatched input/output "
                    f"quantization (in scale={in_t.scale} zp={in_t.zero_point}, "
                    f"out scale={out_t.scale} zp={out_t.zero_point}); EdgeForge's pooling kernel "
                    "assumes these match and doesn't rescale."
                )


def render_model(ir: ModelIR, module_name: str = "model") -> str:
    validate_codegen_support(ir)
    env = make_env()
    ctx = {"ir": ir, "module_name": module_name, "edgeforge_version": __version__}
    return env.get_template("model.v.j2").render(**ctx)


def _quantize_samples(ir: ModelIR, samples: np.ndarray) -> np.ndarray:
    """Raw real-valued sample rows -> the same integer representation the generated RTL's
    input port carries: Q16.16 for a float32 (classical-tier) input tensor, TFLite-style
    int8 for a quantized (deep-tier) one -- so both the RTL and the Python-side reference
    compare the exact same rounding of the exact same input."""
    spec = ir.input_spec
    if spec.dtype == "float32":
        return to_fixed_array(samples)
    scale, zero_point = spec.scale[0], spec.zero_point[0]
    quantize = np.vectorize(lambda x: quantize_affine(float(x), scale, zero_point))
    return quantize(samples).astype(np.int64)


def render_testbench(ir: ModelIR, samples: np.ndarray, module_name: str = "model") -> str:
    """`samples`: an (N, n_in) array of RAW (real-valued, not yet quantized) sample inputs."""
    env = make_env()
    n_samples, n_in = samples.shape
    fixed = _quantize_samples(ir, samples)
    lines = [
        f"        ef_sample[{s * n_in + b}] = {verilog_signed_literal(int(fixed[s, b]))};"
        for s in range(n_samples)
        for b in range(n_in)
    ]
    ctx = {
        "ir": ir,
        "module_name": module_name,
        "edgeforge_version": __version__,
        "n_samples": n_samples,
        "sample_init_lines": lines,
    }
    return env.get_template("testbench.v.j2").render(**ctx)
