"""Verilog source formatting helpers for the classical-tier codegen.

Every IR tensor becomes one flat (row-major) 1-D `reg` array, indexed by a manually
computed flat offset at each use site (see ops.v.j2) -- simpler than mirroring the C
codegen's nested multi-dimensional array declarations, and just as correct since both
are the same underlying row-major memory layout.
"""

from __future__ import annotations

from edgeforge.codegen.formatting import sanitize_ident
from edgeforge.ir import TensorSpec
from edgeforge.verilog.fixedpoint import to_fixed_array, verilog_signed_literal


def buf_name(tensor_name: str) -> str:
    return "ef_" + sanitize_ident(tensor_name)


def tensor_init_lines(tensor: TensorSpec, indent: str = "        ") -> list[str]:
    """`<name>[i] = <literal>;` lines for an `initial` block, one per flattened element.
    A float32 tensor (weights/thresholds/leaf values) is Q16.16-quantized; an int32/uint32
    tensor (a tree's feature-index/child-index arrays) is used as-is -- it's already a plain
    integer, not a real value, so scaling it into Q16.16 would corrupt it."""
    name = buf_name(tensor.name)
    if tensor.dtype == "float32":
        flat = to_fixed_array(tensor.data).reshape(-1)
    elif tensor.dtype in ("int32", "uint32", "int8", "uint8"):
        flat = tensor.data.reshape(-1).astype(int)
    else:
        raise ValueError(f"tensor '{tensor.name}': no Verilog init mapping for dtype {tensor.dtype!r}")
    return [f"{indent}{name}[{i}] = {verilog_signed_literal(v)};" for i, v in enumerate(flat.tolist())]
