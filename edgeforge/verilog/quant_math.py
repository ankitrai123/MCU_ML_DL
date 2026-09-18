"""Integer (multiply + shift) approximation of a float requantization multiplier, for the
deep-tier (int8/TFLite) Verilog backend -- the standard technique real quantized-NN hardware
uses, and the one TFLite Micro itself falls back to on a target with no float unit. See
edgeforge/codegen/quant_math.py's docstring for why the *C* codegen doesn't need this: every
MCU board it targets has usable float (hardware FPU or libgcc soft-float); a small Lattice
FPGA's RTL genuinely has neither.

`quantize_multiplier(m)` decomposes a real (in practice non-negative -- TFLite's own
requantization multipliers, `in_scale * w_scale / out_scale`, are always positive) float `m`
into `(m0, shift)` such that `m ~= m0 * 2**(shift - 31)`, with `m0` a signed 32-bit "Q0.31"
fixed-point fraction in `[2**30, 2**31)` -- the classic frexp-based decomposition (TFLite's own
QuantizeMultiplier, minus its ARM-NEON-oriented "doubling" convention, which this doesn't need
to match bit-for-bit; only self-consistency between this function and the generated RTL does).

`apply_multiplier(acc, m0, shift)` is the matching pure-Python integer reference -- bit widths
and rounding (round-half-up via adding a half-ULP before an arithmetic right shift, matching
Verilog's `>>>` on a signed value exactly, since both are floor-division-by-power-of-2) are
chosen to match exactly what model.v.j2 computes in RTL, so this is both the codegen's own
numeric derivation *and* the ground truth a unit test checks the RTL against.
"""

from __future__ import annotations

import math


def quantize_multiplier(m: float) -> tuple[int, int]:
    if m == 0.0:
        return 0, 0
    q, shift = math.frexp(m)  # m == q * 2**shift, with 0.5 <= |q| < 1
    m0 = round(q * (1 << 31))
    if m0 == (1 << 31):  # rounded exactly up to 1.0 -- push the extra factor of 2 into shift
        m0 //= 2
        shift += 1
    return m0, shift


def apply_multiplier(acc: int, m0: int, shift: int) -> int:
    """`round(acc * m)` computed the same way the generated RTL does: an integer multiply
    followed by a single (rounding) shift, never a float operation."""
    product = acc * m0
    total_shift = 31 - shift
    if total_shift <= 0:
        return product << (-total_shift)
    return (product + (1 << (total_shift - 1))) >> total_shift


def requant_multipliers_for_node(node, ir) -> list[tuple[int, int]]:
    """Per-output-channel (m0, total_shift) pairs ready for the RTL: reuses the existing
    float multiplier derivation (edgeforge.codegen.quant_math.requant_multipliers, shared
    with the C codegen -- same in_scale*weight_scale/out_scale math) and quantizes each one."""
    from edgeforge.codegen.quant_math import requant_multipliers as _float_multipliers

    pairs = []
    for m in _float_multipliers(node, ir):
        m0, shift = quantize_multiplier(m)
        pairs.append((m0, 31 - shift))
    return pairs
