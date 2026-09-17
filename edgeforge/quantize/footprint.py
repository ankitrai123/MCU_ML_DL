"""Footprint estimation and board-budget gating.

This runs *before* any code is generated: report the parameter count and
estimated flash/RAM cost, and refuse early with a specific, actionable
message if the model won't fit -- the spec's "fail early... instead of
emitting broken code," applied before a single template is rendered.

Flash is estimated from constant (weight/bias) storage; RAM from intermediate
activation scratch buffers plus the input/output buffers. This intentionally
does not try to predict compiled code size (compiler- and optimization-level
dependent, and dominated in practice by weight/activation data for any
nontrivial model) -- each board's ``flash_reserved_bytes``/``ram_reserved_bytes``
headroom is exactly the place that per-architecture runtime/code-size cost is
supposed to live. Once a toolchain actually compiles the output,
``edgeforge.build`` reports the real, measured size, which supersedes this
estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

from edgeforge.boards.schema import BoardProfile
from edgeforge.errors import FootprintExceededError, TierMismatchError
from edgeforge.ir import ModelIR


@dataclass(frozen=True)
class FootprintReport:
    param_count: int
    estimated_flash_bytes: int
    estimated_ram_bytes: int
    available_flash_bytes: int
    available_ram_bytes: int

    @property
    def fits_flash(self) -> bool:
        return self.estimated_flash_bytes <= self.available_flash_bytes

    @property
    def fits_ram(self) -> bool:
        return self.estimated_ram_bytes <= self.available_ram_bytes

    @property
    def fits(self) -> bool:
        return self.fits_flash and self.fits_ram

    def describe(self, board: BoardProfile) -> str:
        def pct(used, avail):
            return f"{100.0 * used / avail:5.1f}%" if avail else "  n/a"

        return (
            f"model: {self.param_count} parameters\n"
            f"  flash (weights):     {self.estimated_flash_bytes:>8} / {self.available_flash_bytes} bytes "
            f"({pct(self.estimated_flash_bytes, self.available_flash_bytes)} of budget on {board.id})\n"
            f"  RAM (activations+io): {self.estimated_ram_bytes:>8} / {self.available_ram_bytes} bytes "
            f"({pct(self.estimated_ram_bytes, self.available_ram_bytes)} of budget on {board.id})"
        )


def estimate_footprint(ir: ModelIR, board: BoardProfile) -> FootprintReport:
    # main.c keeps its own copies of the input/output buffers (raw sensor reading, quantized
    # input, and the model_infer output) alongside model.c's -- roughly another io_bytes()
    # worth of RAM -- so that's counted twice, on top of the internal activation scratch.
    ram_estimate = (ir.activation_bytes() + 2 * ir.io_bytes()) * board.memory.ram_overhead_multiplier
    ram_estimate = int(round(ram_estimate))
    return FootprintReport(
        param_count=ir.param_count(),
        estimated_flash_bytes=ir.weight_bytes(),
        estimated_ram_bytes=ram_estimate,
        available_flash_bytes=board.memory.available_flash_bytes,
        available_ram_bytes=board.memory.available_ram_bytes,
    )


def check_budget(ir: ModelIR, board: BoardProfile) -> FootprintReport:
    """Gate a model against a board: tier compatibility, then flash/RAM budget.
    Raises TierMismatchError / FootprintExceededError with an actionable message; returns
    the report on success so the caller can print it either way."""
    if not board.accepts_tier(ir.kind):
        raise TierMismatchError(
            f"board '{board.id}' is model_tier={board.model_tier!r}, which only accepts classical "
            f"models, but this model is kind={ir.kind!r} (a neural net). Target a model_tier: deep "
            f"board instead, or use a classical model (decision tree / logistic regression / small "
            f"MLP) on '{board.id}'."
        )

    report = estimate_footprint(ir, board)
    if not report.fits_flash:
        raise FootprintExceededError(
            f"model needs an estimated {report.estimated_flash_bytes} bytes of flash for weights, "
            f"but board '{board.id}' only has {report.available_flash_bytes} bytes available "
            f"(flash_bytes={board.memory.flash_bytes} minus flash_reserved_bytes={board.memory.flash_reserved_bytes} "
            f"for runtime/code). Reduce model size (fewer parameters, a shallower tree/network) or "
            f"target a board with more flash."
        )
    if not report.fits_ram:
        raise FootprintExceededError(
            f"model needs an estimated {report.estimated_ram_bytes} bytes of RAM for activations and "
            f"input/output buffers, but board '{board.id}' only has {report.available_ram_bytes} bytes "
            f"available (ram_bytes={board.memory.ram_bytes} minus ram_reserved_bytes={board.memory.ram_reserved_bytes} "
            f"for stack/globals). Reduce hidden-layer width/input size or target a board with more RAM."
        )
    return report
