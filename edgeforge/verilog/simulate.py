"""Host-only correctness verification for the classical-tier Verilog backend.

Compiles the generated model.v + testbench.v with Icarus Verilog (`iverilog`/`vvp`) --
never a synthesis toolchain -- and compares predicted classes / dequantized output values
against the *real* source model via `ingest_result.reference_fn`. This is exactly
golden-vector validation's own "prove correctness on the host before trusting anything
downstream" guarantee, aimed at simulated RTL instead of cross-compiled C.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgeforge.errors import ToolchainNotFoundError, ValidationError
from edgeforge.ingest.base import IngestResult
from edgeforge.verilog import codegen
from edgeforge.verilog.fixedpoint import from_fixed_array

# Q16.16 gives ~1.5e-5 resolution; a few LSBs of rounding slack across a chain of affine/linear
# nodes is expected (the same "quantization-aware tolerance" spirit as golden.py's deep-tier
# check), not a sign of a wrong result.
REGRESSION_ATOL = 1e-2

_DATA_LINE_RE = re.compile(r"^-?\d+(,-?\d+)+$")


@dataclass
class VerilogSample:
    index: int
    input_vec: np.ndarray
    ref_class: "int | None"
    ref_output: np.ndarray
    got_class: "int | None"
    got_output: np.ndarray
    matched: bool
    detail: str


@dataclass
class VerilogSimReport:
    samples: list[VerilogSample]
    n_total: int

    @property
    def n_pass(self) -> int:
        return sum(1 for s in self.samples if s.matched)

    @property
    def all_passed(self) -> bool:
        return self.n_pass == self.n_total

    def describe(self) -> str:
        lines = [f"Verilog simulation (iverilog): {self.n_pass}/{self.n_total} samples matched"]
        for s in self.samples:
            if not s.matched:
                lines.append(f"  sample {s.index}: MISMATCH -- {s.detail}")
        return "\n".join(lines)


def run_verilog_simulation(
    ingest_result: IngestResult,
    out_dir: Path,
    module_name: str = "model",
    n_samples: int = 20,
    seed: int = 0,
) -> VerilogSimReport:
    ir = ingest_result.ir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        raise ToolchainNotFoundError(
            "Verilog simulation needs Icarus Verilog ('iverilog'/'vvp'), which isn't installed "
            "(or not on PATH). This always runs on the host, regardless of target FPGA -- it's "
            "the RTL equivalent of golden-vector validation's host C compiler, not the ECP5 "
            "synthesis toolchain. On Linux: apt-get install iverilog. On macOS: brew install "
            "icarus-verilog."
        )

    rng = np.random.default_rng(seed)
    inputs = np.asarray(ingest_result.input_sampler(rng, n_samples), dtype=np.float64)
    if inputs.shape[0] != n_samples:
        raise ValidationError(f"input_sampler returned {inputs.shape[0]} samples, expected {n_samples}")

    refs = [ingest_result.reference_fn(inputs[i]) for i in range(n_samples)]

    model_v_path = out_dir / f"{module_name}.v"
    tb_v_path = out_dir / "testbench.v"
    model_v_path.write_text(codegen.render_model(ir, module_name=module_name))
    tb_v_path.write_text(codegen.render_testbench(ir, inputs, module_name=module_name))

    sim_bin = out_dir / "sim.out"
    compile_cmd = ["iverilog", "-g2012", "-o", str(sim_bin), str(model_v_path), str(tb_v_path)]
    compile_res = subprocess.run(compile_cmd, cwd=str(out_dir), capture_output=True, text=True, timeout=60)
    if compile_res.returncode != 0:
        raise ValidationError(
            f"generated Verilog failed to compile with iverilog:\n$ {' '.join(compile_cmd)}\n"
            f"{compile_res.stdout}{compile_res.stderr}"
        )

    try:
        run_res = subprocess.run(["vvp", str(sim_bin)], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as e:
        raise ValidationError("Verilog simulation timed out") from e
    if run_res.returncode != 0:
        raise ValidationError(f"Verilog simulation crashed (exit {run_res.returncode}): {run_res.stderr}")

    # vvp prints its own "$finish called at ..." diagnostic to stdout after the testbench's
    # real output -- filter to lines that are actually our comma-separated-integers format
    # rather than assuming stdout is otherwise clean.
    lines = [ln for ln in run_res.stdout.strip().splitlines() if _DATA_LINE_RE.match(ln.strip())]
    if len(lines) != n_samples:
        raise ValidationError(
            f"expected {n_samples} output lines from the Verilog simulation, got {len(lines)}:\n{run_res.stdout}"
        )

    samples = []
    for i, line in enumerate(lines):
        parts = line.split(",")
        got_class = int(parts[0])
        got_class = None if got_class == -1 else got_class
        got_output = from_fixed_array(int(x) for x in parts[1:])
        ref_class, ref_output = refs[i]

        if ir.task == "classification":
            matched = got_class == ref_class
            detail = f"Verilog predicted class {got_class}, Python reference predicted class {ref_class}"
        else:
            matched = bool(np.allclose(got_output, ref_output, atol=REGRESSION_ATOL))
            detail = (
                f"Verilog output {got_output.tolist()} vs Python reference {ref_output.tolist()} "
                f"(atol={REGRESSION_ATOL})"
            )

        samples.append(
            VerilogSample(
                index=i,
                input_vec=inputs[i],
                ref_class=ref_class,
                ref_output=ref_output,
                got_class=got_class,
                got_output=got_output,
                matched=matched,
                detail=detail,
            )
        )

    return VerilogSimReport(samples=samples, n_total=n_samples)
