"""Renders a ModelIR + BoardProfile into a directory of compilable C source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from edgeforge.boards.schema import BoardProfile
from edgeforge.codegen.context import build_context, make_env
from edgeforge.ir import ModelIR


@dataclass
class GeneratedFiles:
    out_dir: Path
    model_h: Path
    model_c: Path
    main_c: Path
    linker_script: Optional[Path]
    startup_c: Optional[Path]

    def sources(self) -> list[Path]:
        """.c files to hand the compiler for the board's own firmware build (model.c + main.c
        + startup.c if the board needs one) -- excludes golden_test.c, which validate/golden.py
        compiles separately against a host-only harness."""
        srcs = [self.model_c, self.main_c]
        if self.startup_c:
            srcs.append(self.startup_c)
        return srcs


def generate(ir: ModelIR, board: BoardProfile, out_dir: Path, model_name: str = "model") -> GeneratedFiles:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env()
    ctx = build_context(ir, board, model_name)

    model_h_path = out_dir / f"{model_name}.h"
    model_h_path.write_text(env.get_template("model.h.j2").render(**ctx))

    model_c_path = out_dir / f"{model_name}.c"
    model_c_path.write_text(env.get_template("model.c.j2").render(**ctx))

    main_c_path = out_dir / "main.c"
    main_c_path.write_text(env.get_template("main.c.j2").render(**ctx))

    linker_script_path = None
    if board.toolchain.linker_script:
        linker_script_path = out_dir / "link.ld"
        linker_script_path.write_text(env.get_template(f"linker/{board.toolchain.linker_script}").render(**ctx))

    startup_c_path = None
    if board.toolchain.needs_startup_stub:
        startup_c_path = out_dir / "startup.c"
        startup_c_path.write_text(env.get_template("startup.c.j2").render(**ctx))

    return GeneratedFiles(
        out_dir=out_dir,
        model_h=model_h_path,
        model_c=model_c_path,
        main_c=main_c_path,
        linker_script=linker_script_path,
        startup_c=startup_c_path,
    )
