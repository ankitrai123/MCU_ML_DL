"""Q16.16 fixed-point conversion for the classical-tier (float32 IR) Verilog backend.

A small Lattice FPGA has no hardware floating-point unit, and building one in RTL is a much
bigger undertaking than any classical-tier model (a decision tree, a logistic-regression-style
linear layer, a folded sklearn scaler) needs -- so every classical-tier value gets a single,
fixed signed 32-bit Q16.16 format instead: 16 integer bits (including sign) and 16 fractional
bits, +-32768 range at 1/65536 (~1.5e-5) resolution. That comfortably covers any sklearn
model's weights/thresholds and typical sensor-scale feature values, so (unlike the deep tier's
per-tensor TFLite-style scale/zero-point -- see quant_math.py) one shared constant is enough:
no per-tensor scale computation, and every signal in the generated RTL is the same width and
format. Comparisons (decision tree thresholds) are correct after quantization because Q16.16
is a monotonic (order-preserving) transform of the real value; multiply-accumulate needs one
extra step (see model.v.j2's use of this module) to renormalize a product back to Q16.16.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

FRACTIONAL_BITS = 16
SCALE = 1 << FRACTIONAL_BITS  # 65536
WIDTH = 32
QMIN = -(1 << (WIDTH - 1))
QMAX = (1 << (WIDTH - 1)) - 1


def to_fixed(x: float) -> int:
    """Quantize one real value to a Q16.16 signed 32-bit int, clamping on overflow
    rather than silently wrapping (a model weight/threshold this far out of range
    would be a red flag, not something to hide)."""
    q = int(round(float(x) * SCALE))
    return max(QMIN, min(QMAX, q))


def to_fixed_array(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    q = np.round(a * SCALE)
    return np.clip(q, QMIN, QMAX).astype(np.int64)


def from_fixed(q: int) -> float:
    return float(q) / SCALE


def from_fixed_array(arr: Iterable[int]) -> np.ndarray:
    return np.asarray(list(arr), dtype=np.float64) / SCALE


def fixed_literal(x: float, width: int = WIDTH) -> str:
    """Convenience: quantize and format in one step, for a constant that appears directly
    in a template (e.g. relu6's 6.0 ceiling) rather than coming from a tensor's own data."""
    return verilog_signed_literal(to_fixed(x), width)


def verilog_signed_literal(q: int, width: int = WIDTH) -> str:
    """`-32'sd6789`-style signed decimal literal Verilog accepts directly (avoids
    two's-complement hex juggling, and reads back as the same value in generated source)."""
    q = int(q)
    if q < 0:
        return f"-{width}'sd{-q}"
    return f"{width}'sd{q}"
