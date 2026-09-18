"""Renders a ModelIR + BoardProfile into a directory of compilable C source (or, for a
board whose toolchain.kind is "arduino-cli", an arduino-cli-compilable sketch folder)."""

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
    main_c: Optional[Path]
    linker_script: Optional[Path]
    startup_c: Optional[Path]
    sketch_ino: Optional[Path] = None  # set instead of main_c for an arduino-cli board

    def sources(self) -> list[Path]:
        """.c files to hand a *raw* cross-compiler for the board's own firmware build (model.c
        + main.c + startup.c if the board needs one) -- excludes golden_test.c, which
        validate/golden.py compiles separately against a host-only harness. Unused for an
        arduino-cli board: arduino-cli compiles every file already sitting in the sketch
        directory itself, not a flag-driven source list."""
        srcs = [s for s in (self.model_c, self.main_c) if s is not None]
        if self.startup_c:
            srcs.append(self.startup_c)
        return srcs


def generate(ir: ModelIR, board: BoardProfile, out_dir: Path, model_name: str = "model") -> GeneratedFiles:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env()
    ctx = build_context(ir, board, model_name)

    if board.toolchain.kind == "arduino-cli":
        return _generate_arduino_sketch(env, ctx, out_dir, model_name)

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


def _generate_arduino_sketch(env, ctx: dict, out_dir: Path, model_name: str) -> GeneratedFiles:
    """arduino-cli/the Arduino IDE require a sketch's .ino to share its containing directory's
    name, so everything lives in out_dir/<model_name>_sketch/ instead of flat in out_dir the
    way the raw-toolchain boards above lay their files out. model.h/model.c render unchanged
    (see model.h.j2's extern "C" guard) and just live alongside the .ino -- arduino-cli
    compiles every source file in a sketch directory together automatically."""
    sketch_name = f"{model_name}_sketch"
    sketch_dir = out_dir / sketch_name
    sketch_dir.mkdir(parents=True, exist_ok=True)

    model_h_path = sketch_dir / f"{model_name}.h"
    model_h_path.write_text(env.get_template("model.h.j2").render(**ctx))

    model_c_path = sketch_dir / f"{model_name}.c"
    model_c_path.write_text(env.get_template("model.c.j2").render(**ctx))

    sketch_ino_path = sketch_dir / f"{sketch_name}.ino"
    sketch_ino_path.write_text(env.get_template("sketch.ino.j2").render(**ctx))

    return GeneratedFiles(
        out_dir=out_dir,
        model_h=model_h_path,
        model_c=model_c_path,
        main_c=None,
        linker_script=None,
        startup_c=None,
        sketch_ino=sketch_ino_path,
    )
