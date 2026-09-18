"""Keras model -> FPGA-targeted HLS project, via hls4ml (Phase 3, initial version).

Unlike every board in edgeforge/boards/, this deliberately does NOT go through ir.py or
edgeforge/codegen/ at all: hls4ml performs its own independent conversion from the *original*
Keras model object into HLS C++ (per-layer templates from its own nnet_utils library), which
is a fundamentally different kind of output -- a synthesizable hardware description, not a
CPU-executable program -- from anything else this project generates. That split matches the
roadmap's own framing ("wrapping hls4ml/FINN rather than writing HLS generation from
scratch"), so this module is a thin orchestration layer around hls4ml, not a new code
generator.

It does, however, carry over the one guarantee every other board gets: a host-only
correctness check before you trust the output. hls4ml's own C-simulation -- a plain
g++-compiled shared library, confirmed to need no FPGA vendor toolchain at all (unlike
`ModelGraph.build()`, which hard-requires a real synthesis tool on PATH -- `vivado_hls` for
the default Vivado backend, a different tool for each other backend hls4ml supports, see
_BACKEND_TOOLCHAINS below) -- is run here and compared against the original Keras model's
predictions, exactly like golden-vector validation elsewhere in EdgeForge. Real synthesis
(actual LUT/FF/DSP/BRAM numbers, an exportable bitstream) is a separate, best-effort step
gated on that backend's own toolchain being installed, mirroring how a missing target
cross-compiler degrades gracefully for every MCU board. `--backend` accepts any backend
hls4ml itself supports (confirmed working here: Vivado, Vitis, Quartus, Catapult) -- csim and
host-verification work identically regardless, since every backend's C-simulation is still
just a plain g++-compiled shared library.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from edgeforge.errors import EdgeForgeError

DEFAULT_PART = "xc7z020clg400-1"  # Zynq-7020 -- e.g. PYNQ-Z2 / Zybo Z7-20, hls4ml's own common default
DEFAULT_BACKEND = "Vivado"
DEFAULT_PRECISION = "fixed<16,6>"
DEFAULT_SAMPLE_RANGE = (0.0, 1.0)  # matches keras_ingest.py's own default when no representative data is given

# Every hls4ml backend shells out to its own real synthesis tool, under a *different* name and
# with a *different* ModelGraph.build() call signature -- confirmed by reading each backend's
# own build() source, not assumed from one (Vivado's). A single hardcoded "check for vivado_hls,
# call build(csim=False, synth=True, export=False)" is wrong for every other backend: it reports
# the wrong missing tool, and (Quartus/OneAPI don't accept csim/synth/export at all) can raise a
# bare TypeError instead of ever reaching hls4ml's own, more useful error message.
_BACKEND_TOOLCHAINS = {
    "vivado": ("vivado_hls", "Install Xilinx Vivado HLS/Vitis HLS (https://www.xilinx.com/support/download.html) and add vivado_hls to PATH."),
    "vitis": ("vitis-run", "Install Xilinx Vitis HLS (https://www.xilinx.com/support/download.html) and add vitis-run to PATH."),
    "quartus": ("i++", "Install Intel Quartus Prime / HLS Compiler (https://www.intel.com/content/www/us/en/software-kit/661750/) and add i++ to PATH."),
    "catapult": ("catapult", "Install Siemens Catapult HLS (https://eda.sw.siemens.com/en-US/ic/catapult-high-level-synthesis/) and add catapult to PATH."),
    "oneapi": ("icpx", "Install the Intel oneAPI DPC++/C++ Compiler (https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) and add icpx to PATH."),
}
# build() kwargs per backend -- only the ones every backend in _BACKEND_TOOLCHAINS actually
# accepts (confirmed per-backend above); Quartus/OneAPI take none of these, so they get {}.
_BACKEND_BUILD_KWARGS = {
    "vivado": {"csim": False, "synth": True, "export": False},
    "vitis": {"csim": False, "synth": True, "export": False},
    "catapult": {"csim": False, "synth": True, "export": False},
    "quartus": {"synth": True},
    "oneapi": {},
}


class HlsToolNotFoundError(EdgeForgeError):
    """hls4ml itself, or a Keras-loading dependency (TensorFlow), is not installed."""


@dataclass
class FpgaSample:
    index: int
    input_vec: np.ndarray
    ref_output: np.ndarray
    hls_output: np.ndarray
    matched: bool
    max_abs_diff: float


@dataclass
class FpgaValidationReport:
    samples: list[FpgaSample]
    n_total: int
    tolerance: float

    @property
    def n_pass(self) -> int:
        return sum(1 for s in self.samples if s.matched)

    @property
    def all_passed(self) -> bool:
        return self.n_pass == self.n_total

    def describe(self) -> str:
        lines = [f"HLS C-simulation validation: {self.n_pass}/{self.n_total} samples within tolerance ({self.tolerance})"]
        for s in self.samples:
            if not s.matched:
                lines.append(f"  sample {s.index}: max abs diff {s.max_abs_diff:.5f} exceeds tolerance")
        return "\n".join(lines)


@dataclass
class FpgaBuildResult:
    ok: bool
    output_dir: Optional[Path] = None
    param_count: int = 0
    validation: Optional[FpgaValidationReport] = None
    synth_attempted: bool = False
    synth_ok: bool = False
    synth_report: Optional[dict] = None
    toolchain_missing: Optional[str] = None
    install_hint: str = ""
    error: Optional[str] = None


def _load_keras_model(model_path: Path) -> Any:
    try:
        import tensorflow as tf
    except ImportError as e:
        raise HlsToolNotFoundError(
            f"the FPGA backend needs TensorFlow to load a Keras model, which is not installed ({e}). "
            "Install it with: pip install tensorflow-cpu"
        ) from None
    return tf.keras.models.load_model(model_path)


def _require_hls4ml() -> Any:
    try:
        import hls4ml
    except ImportError as e:
        raise HlsToolNotFoundError(
            f"the FPGA backend needs the 'hls4ml' package, which is not installed ({e}). "
            "Install it with: pip install hls4ml"
        ) from None
    return hls4ml


def convert_and_validate(
    model_path: Path,
    out_dir: Path,
    part: str = DEFAULT_PART,
    backend: str = DEFAULT_BACKEND,
    precision: str = DEFAULT_PRECISION,
    reuse_factor: int = 1,
    sample_range: tuple[float, float] = DEFAULT_SAMPLE_RANGE,
    n_samples: int = 20,
    seed: int = 0,
    tolerance: float = 0.05,
    attempt_synthesis: bool = False,
) -> FpgaBuildResult:
    hls4ml = _require_hls4ml()
    model = _load_keras_model(model_path)
    n_features = model.input_shape[-1]
    param_count = model.count_params()
    out_dir = Path(out_dir)

    config = hls4ml.utils.config_from_keras_model(
        model, granularity="model", default_precision=precision, default_reuse_factor=reuse_factor
    )
    hmodel = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=str(out_dir), part=part, backend=backend
    )
    hmodel.write()

    try:
        hmodel.compile()
    except Exception as e:
        return FpgaBuildResult(
            ok=False, output_dir=out_dir, param_count=param_count,
            error=f"HLS C-simulation build failed (host g++, no FPGA toolchain involved): {e}",
        )

    rng = np.random.default_rng(seed)
    lo, hi = sample_range
    X = rng.uniform(lo, hi, size=(n_samples, n_features)).astype(np.float32)
    ref = np.asarray(model.predict(X, verbose=0))
    got = np.asarray(hmodel.predict(X))

    samples = []
    for i in range(n_samples):
        diff = float(np.max(np.abs(got[i] - ref[i])))
        samples.append(FpgaSample(
            index=i, input_vec=X[i], ref_output=ref[i], hls_output=got[i],
            matched=diff <= tolerance, max_abs_diff=diff,
        ))
    validation = FpgaValidationReport(samples=samples, n_total=n_samples, tolerance=tolerance)
    result = FpgaBuildResult(ok=validation.all_passed, output_dir=out_dir, param_count=param_count, validation=validation)

    if attempt_synthesis:
        backend_key = backend.lower()
        toolchain_info = _BACKEND_TOOLCHAINS.get(backend_key)
        if toolchain_info is not None:
            tool, install_hint = toolchain_info
            if shutil.which(tool) is None:
                result.toolchain_missing = tool
                result.install_hint = f"{install_hint} Then re-run with --synthesize."
            else:
                result.synth_attempted = True
                try:
                    result.synth_report = hmodel.build(**_BACKEND_BUILD_KWARGS[backend_key])
                    result.synth_ok = True
                except Exception as e:
                    result.error = f"{backend} synthesis failed: {e}"
        else:
            # An hls4ml backend we haven't mapped a toolchain binary for (e.g. VivadoAccelerator,
            # Libero) -- rather than guess a binary name and risk reporting the wrong tool as
            # missing, attempt the build with that backend's own defaults and surface whatever
            # real error it raises.
            result.synth_attempted = True
            try:
                result.synth_report = hmodel.build()
                result.synth_ok = True
            except Exception as e:
                result.error = f"{backend} synthesis failed: {e}"

    return result
