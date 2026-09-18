"""Renders synthesizable Verilog + a host-simulation testbench from a classical-tier ModelIR.

Restricted, in this increment, to tree/linear(float32 weights)/affine -- exactly the ops the
sklearn ingest path produces. A conv2d/depthwise_conv2d/maxpool2d/int8-quantized-linear node
(the deep/TFLite tier) raises a clear error naming what's not supported yet, rather than
silently emitting wrong RTL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from edgeforge import __version__
from edgeforge.errors import UnsupportedOpError
from edgeforge.ir import ModelIR
from edgeforge.verilog import formatting
from edgeforge.verilog.fixedpoint import fixed_literal, to_fixed_array, verilog_signed_literal

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

CLASSICAL_OPS = ("tree", "linear", "affine", "activation")
_CLASSICAL_ACTIVATION_KINDS = ("relu", "relu6", "identity")


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
    )
    return env


def validate_classical_tier(ir: ModelIR) -> None:
    """Raises UnsupportedOpError naming exactly what's missing, rather than letting an
    unhandled node silently fall through Jinja2's StrictUndefined into a confusing template
    error (or, worse, an empty/wrong RTL block)."""
    for node in ir.nodes:
        if node.op not in CLASSICAL_OPS:
            raise UnsupportedOpError(
                f"node '{node.name}' (op={node.op!r}): the Verilog/Lattice backend's classical tier "
                f"only supports {', '.join(CLASSICAL_OPS)} so far -- deep-tier ops "
                "(conv2d/depthwise_conv2d/maxpool2d/int8-quantized linear/activation) aren't "
                "implemented yet."
            )
        if node.op == "linear":
            w = ir.tensors[node.attrs["weight"]]
            if w.dtype != "float32":
                raise UnsupportedOpError(
                    f"node '{node.name}': a quantized ({w.dtype}) linear layer isn't supported by "
                    "the Verilog backend's classical tier yet -- only float32 (sklearn-style) "
                    "linear layers are."
                )
        if node.op == "activation" and node.attrs.get("kind") not in _CLASSICAL_ACTIVATION_KINDS:
            raise UnsupportedOpError(
                f"node '{node.name}': activation kind {node.attrs.get('kind')!r} needs a "
                f"transcendental function EdgeForge doesn't emit in hardware for any backend -- "
                f"only {', '.join(_CLASSICAL_ACTIVATION_KINDS)} are supported."
            )


def render_model(ir: ModelIR, module_name: str = "model") -> str:
    validate_classical_tier(ir)
    env = make_env()
    ctx = {"ir": ir, "module_name": module_name, "edgeforge_version": __version__}
    return env.get_template("model.v.j2").render(**ctx)


def render_testbench(ir: ModelIR, samples: np.ndarray, module_name: str = "model") -> str:
    """`samples`: an (N, n_in) array of RAW (real-valued, not yet quantized) sample inputs --
    quantized here, once, so both the RTL and the Python-side reference compare the exact
    same Q16.16 rounding of the same input, the same way golden-vector validation compares
    the exact same raw sample row against the reference model and the generated C."""
    env = make_env()
    n_samples, n_in = samples.shape
    fixed = to_fixed_array(samples)
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
