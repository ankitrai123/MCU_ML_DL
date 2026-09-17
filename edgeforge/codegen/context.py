"""Jinja2 environment + per-render template context.

Every value a template needs is either plain IR/board data or one of the
helper functions below -- nothing here or in any ``.j2`` file branches on
``board.id``. A board changes behavior only by changing the *values* these
templates substitute (a type string, a qualifier, a toolchain flag).
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from edgeforge import __version__
from edgeforge.boards.schema import BoardProfile
from edgeforge.codegen import formatting, quant_math
from edgeforge.ir import ModelIR

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

_TRANSCENDENTAL_ACTIVATIONS = {"softmax", "sigmoid", "logistic", "tanh"}


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
        c_array_literal=formatting.c_array_literal,
        c_dims=formatting.c_dims,
        float_literal=formatting.float_literal,
        int_literal=formatting.int_literal,
        board_ctype_for=formatting.board_ctype_for,
        tensor_ref=formatting.tensor_ref,
        flat_ref=formatting.flat_ref,
        hex32=formatting.hex32,
        requant_multipliers=quant_math.requant_multipliers,
        fused_activation_bounds=quant_math.fused_activation_bounds,
    )
    return env


def validate_codegen_support(ir: ModelIR) -> None:
    """Catch an ingest-layer oversight before it becomes silently-wrong C: no op
    that needs a transcendental function should ever reach codegen (every ingester
    is responsible for eliding a final softmax/sigmoid or rejecting a mid-graph one)."""
    from edgeforge.errors import UnsupportedOpError

    for node in ir.nodes:
        if node.op == "activation" and node.attrs.get("kind") in _TRANSCENDENTAL_ACTIVATIONS:
            raise UnsupportedOpError(
                f"internal error: node '{node.name}' (activation kind={node.attrs['kind']!r}) reached "
                "codegen without being elided or rejected at ingest -- this is an EdgeForge bug, not a "
                "model problem; please report it."
            )
        if node.op == "maxpool2d":
            # The pooling kernel compares/averages raw quantized values directly (no rescale step),
            # which is only valid when TFLite assigned pooling the same in/out quantization params
            # -- true for MAX_POOL_2D always and AVERAGE_POOL_2D in every case this converter path
            # produces, but checked explicitly so a violation is a clear error, not silently wrong math.
            in_t = ir.tensors[node.inputs[0]]
            out_t = ir.tensors[node.outputs[0]]
            if in_t.scale != out_t.scale or in_t.zero_point != out_t.zero_point:
                raise UnsupportedOpError(
                    f"internal error: pooling node '{node.name}' has mismatched input/output "
                    f"quantization (in scale={in_t.scale} zp={in_t.zero_point}, "
                    f"out scale={out_t.scale} zp={out_t.zero_point}); EdgeForge's pooling kernel "
                    "assumes these match and doesn't rescale."
                )


def build_context(ir: ModelIR, board: BoardProfile, model_name: str) -> dict:
    validate_codegen_support(ir)
    return {
        "ir": ir,
        "board": board,
        "model_name": model_name,
        "edgeforge_version": __version__,
        "input_ctype": formatting.board_ctype_for(ir.input_spec, board),
        "output_ctype": board.type_of("real"),
        "class_ctype": board.type_of("class_index"),
        "index_ctype": board.type_of("index"),
    }
