"""Golden-vector validation: the main defense against silent precision bugs.

N sample inputs go through the *original* framework model (sklearn/.predict,
the TFLite Interpreter, onnxruntime) and through the generated C, compiled
into a host-side test harness (never flashed to hardware -- this always uses
the host compiler, regardless of which board the source was generated for).
Classifiers are graded on predicted-class agreement (robust to the small
numeric noise argmax doesn't care about); regressors are graded on numeric
closeness with a quantization-aware tolerance.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgeforge.boards.schema import BoardProfile
from edgeforge.build.toolchain import build_host_executable
from edgeforge.codegen.context import build_context, make_env
from edgeforge.codegen.formatting import float_literal
from edgeforge.errors import ValidationError
from edgeforge.ingest.base import IngestResult
from edgeforge.ir import ModelIR


@dataclass
class GoldenSample:
    index: int
    input_vec: np.ndarray
    ref_class: int | None
    ref_output: np.ndarray
    got_class: int | None
    got_output: np.ndarray
    matched: bool
    detail: str


@dataclass
class GoldenReport:
    samples: list[GoldenSample]
    n_total: int

    @property
    def n_pass(self) -> int:
        return sum(1 for s in self.samples if s.matched)

    @property
    def all_passed(self) -> bool:
        return self.n_pass == self.n_total

    def describe(self) -> str:
        lines = [f"golden-vector validation: {self.n_pass}/{self.n_total} samples matched"]
        for s in self.samples:
            if not s.matched:
                lines.append(f"  sample {s.index}: MISMATCH -- {s.detail}")
        return "\n".join(lines)


def _tolerance_for(ir: ModelIR) -> tuple[float, float]:
    """(rtol, atol) for comparing regression output values -- unused for classification,
    which is graded on class-index equality instead (see module docstring)."""
    if ir.kind == "deep":
        scale = ir.output_spec.scale[0] if ir.output_spec.scale else 1.0
        return 0.0, 2.0 * scale  # 2 quantization steps of slack
    return 1e-3, 1e-4


def run_golden_validation(
    ingest_result: IngestResult,
    board: BoardProfile,
    model_c_path: Path,
    out_dir: Path,
    model_name: str = "model",
    n_samples: int = 20,
    seed: int = 0,
) -> GoldenReport:
    ir = ingest_result.ir
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    inputs = np.asarray(ingest_result.input_sampler(rng, n_samples), dtype=np.float64)
    if inputs.shape[0] != n_samples:
        raise ValidationError(f"input_sampler returned {inputs.shape[0]} samples, expected {n_samples}")

    refs = [ingest_result.reference_fn(inputs[i]) for i in range(n_samples)]

    env = make_env()
    ctx = build_context(ir, board, model_name)
    ctx["n_samples"] = n_samples
    ctx["rows"] = [[float_literal(v) for v in inputs[i]] for i in range(n_samples)]
    golden_c_path = out_dir / "golden_test.c"
    golden_c_path.write_text(env.get_template("golden_test.c.j2").render(**ctx))

    build_res = build_host_executable([model_c_path, golden_c_path], out_dir, board, exe_name="golden_test")
    if not build_res.success:
        raise ValidationError(f"golden-vector harness failed to compile/link on the host:\n{build_res.log}")

    try:
        run_res = subprocess.run([str(build_res.artifact_path)], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as e:
        raise ValidationError("golden-vector harness timed out") from e
    if run_res.returncode != 0:
        raise ValidationError(
            f"golden-vector harness crashed (exit {run_res.returncode}): {run_res.stderr}"
        )

    lines = [ln for ln in run_res.stdout.strip().splitlines() if ln.strip()]
    if len(lines) != n_samples:
        raise ValidationError(
            f"expected {n_samples} output lines from the golden-vector harness, got {len(lines)}:\n{run_res.stdout}"
        )

    rtol, atol = _tolerance_for(ir)
    samples = []
    for i, line in enumerate(lines):
        parts = line.split(",")
        got_class = int(parts[0])
        got_class = None if got_class == -1 else got_class
        got_output = np.array([float(x) for x in parts[1:]], dtype=np.float64)
        ref_class, ref_output = refs[i]

        if ir.task == "classification":
            matched = got_class == ref_class
            detail = f"C predicted class {got_class}, Python reference predicted class {ref_class}"
        else:
            matched = bool(np.allclose(got_output, ref_output, rtol=rtol, atol=atol))
            detail = (
                f"C output {got_output.tolist()} vs Python reference {ref_output.tolist()} "
                f"(rtol={rtol}, atol={atol})"
            )

        samples.append(
            GoldenSample(
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

    return GoldenReport(samples=samples, n_total=n_samples)
