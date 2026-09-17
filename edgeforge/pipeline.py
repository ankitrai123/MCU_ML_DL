"""Orchestrates ingest -> footprint check -> codegen -> build -> validate as one
call, returning a structured result instead of printing/exiting -- shared by the
CLI (`cli.py`) and the local web UI (`webui/`) so neither reimplements this logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from edgeforge.boards.schema import BoardProfile
from edgeforge.build.toolchain import BuildResult, build_firmware
from edgeforge.codegen.generator import GeneratedFiles, generate
from edgeforge.errors import EdgeForgeError, ToolchainNotFoundError
from edgeforge.ingest import detect_and_ingest
from edgeforge.ingest.base import IngestResult
from edgeforge.quantize.footprint import FootprintReport, check_budget
from edgeforge.validate.golden import GoldenReport, run_golden_validation


@dataclass
class ConversionResult:
    ok: bool
    error: Optional[str] = None
    error_stage: Optional[str] = None  # 'ingest' | 'footprint' | 'validate'
    ingest_result: Optional[IngestResult] = None
    footprint: Optional[FootprintReport] = None
    generated: Optional[GeneratedFiles] = None
    build: Optional[BuildResult] = None
    golden: Optional[GoldenReport] = None
    validate_skipped_reason: Optional[str] = None  # set instead of error/error_stage when
    # validation couldn't run because no host C compiler is installed -- not the model's
    # fault and not fatal, exactly like build.toolchain_missing for a target board's compiler.


def run_conversion(
    model_path: Path,
    board: BoardProfile,
    out_dir: Path,
    *,
    model_name: str = "model",
    task: str = "auto",
    sample_range: Optional[tuple[float, float]] = None,
    representative_data: Optional[np.ndarray] = None,
    do_build: bool = True,
    do_validate: bool = True,
    samples: int = 20,
    seed: int = 0,
) -> ConversionResult:
    try:
        ingest_result = detect_and_ingest(
            model_path, sample_range=sample_range, task=task, representative_data=representative_data
        )
    except EdgeForgeError as e:
        return ConversionResult(ok=False, error=str(e), error_stage="ingest")

    try:
        footprint = check_budget(ingest_result.ir, board)
    except EdgeForgeError as e:
        return ConversionResult(ok=False, error=str(e), error_stage="footprint", ingest_result=ingest_result)

    generated = generate(ingest_result.ir, board, out_dir, model_name=model_name)

    build_res = build_firmware(generated, board) if do_build else None

    golden_res = None
    validate_skipped_reason = None
    if do_validate:
        try:
            golden_res = run_golden_validation(
                ingest_result, board, generated.model_c, out_dir, model_name=model_name, n_samples=samples, seed=seed
            )
        except ToolchainNotFoundError as e:
            validate_skipped_reason = str(e)
        except EdgeForgeError as e:
            return ConversionResult(
                ok=False, error=str(e), error_stage="validate",
                ingest_result=ingest_result, footprint=footprint, generated=generated, build=build_res,
            )

    build_ok = build_res is None or build_res.success or bool(build_res.toolchain_missing)
    validate_ok = golden_res is None or golden_res.all_passed
    return ConversionResult(
        ok=build_ok and validate_ok,
        ingest_result=ingest_result,
        footprint=footprint,
        generated=generated,
        build=build_res,
        golden=golden_res,
        validate_skipped_reason=validate_skipped_reason,
    )
