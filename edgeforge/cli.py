"""python -m edgeforge: model + board -> compiling embedded C (and firmware, best-effort)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from edgeforge import __version__
from edgeforge.boards.registry import BoardRegistry
from edgeforge.errors import EdgeForgeError
from edgeforge.ingest import detect_and_ingest
from edgeforge.pipeline import run_conversion
from edgeforge.quantize.footprint import check_budget


def _parse_range(s: str | None) -> tuple[float, float] | None:
    if s is None:
        return None
    lo, hi = s.split(",")
    return float(lo), float(hi)


def _add_common_ingest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, type=Path, help="path to the trained model (.pkl/.pickle, .h5/.keras, .onnx)")
    p.add_argument("--task", choices=["auto", "classification", "regression"], default="auto", help="override task auto-detection")
    p.add_argument("--sample-range", type=str, default=None, metavar="LO,HI", help="override the default input-sampling range used for golden-vector generation (and, for .h5/.keras without --rep-data, synthetic representative data). If LO is negative, write it as --sample-range=-2,8 (with '='), or argparse mistakes '-2,8' for another flag")
    p.add_argument("--rep-data", type=Path, default=None, help=".npy file of representative input samples for TFLite int8 quantization (.h5/.keras only); if omitted, synthetic data is used and int8 accuracy may suffer (the golden-vector check still validates correctness of the generated C, just not real-world model quality)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edgeforge", description=__doc__)
    p.add_argument("--version", action="version", version=f"edgeforge {__version__}")
    p.add_argument("--boards-dir", type=Path, default=None, help="override the board registry directory (default: <repo>/boards)")
    sub = p.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="model + board -> C source (+ compiled firmware, best-effort) + golden-vector check")
    _add_common_ingest_args(conv)
    conv.add_argument("--board", required=True, help="target board id (see 'list-boards')")
    conv.add_argument("--out", required=True, type=Path, help="output directory for generated source/firmware")
    conv.add_argument("--model-name", default="model", help="base name for model.h/model.c (default: 'model')")
    conv.add_argument("--samples", type=int, default=20, help="number of golden-vector samples to check (default: 20)")
    conv.add_argument("--seed", type=int, default=0, help="RNG seed for golden-vector sample generation")
    conv.add_argument("--no-build", action="store_true", help="skip invoking the board's cross-compiler")
    conv.add_argument("--no-validate", action="store_true", help="skip golden-vector validation (not recommended)")

    insp = sub.add_parser("inspect", help="ingest a model and print its IR + (optionally) footprint against a board, without generating code")
    _add_common_ingest_args(insp)
    insp.add_argument("--board", default=None, help="optional: report footprint against this board's budget")

    lb = sub.add_parser("list-boards", help="list available board ids")

    srv = sub.add_parser("serve", help="run a basic local web UI (upload a model, pick a board, convert) at http://host:port")
    srv.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1, i.e. local machine only)")
    srv.add_argument("--port", type=int, default=5000)
    srv.add_argument("--debug", action="store_true", help="enable Flask's debug/reloader mode (development only)")

    fpga = sub.add_parser(
        "convert-fpga",
        help="Keras model -> FPGA-targeted HLS project via hls4ml (experimental, Phase 3) -- "
        "host-verified, synthesis is best-effort and needs Vivado HLS. A separate path from "
        "'convert': no --board, no boards/*.yaml.",
    )
    fpga.add_argument("--model", required=True, type=Path, help="path to a trained Keras model (.h5/.keras)")
    fpga.add_argument("--out", required=True, type=Path, help="output directory for the generated HLS project")
    fpga.add_argument("--part", default="xc7z020clg400-1", help="target FPGA part (default: Zynq-7020, as used on the PYNQ-Z2/Zybo Z7-20)")
    fpga.add_argument("--backend", default="Vivado", help="hls4ml backend (default: Vivado)")
    fpga.add_argument("--precision", default="fixed<16,6>", help="default fixed-point precision for weights/activations (default: fixed<16,6>)")
    fpga.add_argument("--reuse-factor", type=int, default=1, help="hls4ml reuse factor: higher trades latency/throughput for fewer resources (default: 1)")
    fpga.add_argument("--sample-range", type=str, default=None, metavar="LO,HI", help="input sampling range for C-simulation verification (default: 0,1)")
    fpga.add_argument("--samples", type=int, default=20, help="number of C-simulation verification samples (default: 20)")
    fpga.add_argument("--seed", type=int, default=0, help="RNG seed for verification sample generation")
    fpga.add_argument("--tolerance", type=float, default=0.05, help="max abs output difference allowed between the HLS C-simulation and the original Keras model (default: 0.05; loosen for lower precision, tighten for higher)")
    fpga.add_argument("--synthesize", action="store_true", help="also attempt real Vivado HLS synthesis (needs vivado_hls on PATH); without this flag, only the HLS project + host C-simulation verification run")

    vlog = sub.add_parser(
        "convert-verilog",
        help="model -> synthesizable Verilog for the Lattice ECP5 family (experimental, Phase 3) -- "
        "a from-scratch RTL generator (classical tier only so far: tree/linear/affine), "
        "host-verified via Icarus Verilog simulation, synthesis is best-effort via Yosys/NextPNR. "
        "A separate path from 'convert': no --board, no boards/*.yaml.",
    )
    vlog.add_argument("--model", required=True, type=Path, help="path to a trained model (.pkl/.pickle: sklearn tree/logistic-regression/scaler-pipeline only in this increment)")
    vlog.add_argument("--out", required=True, type=Path, help="output directory for the generated Verilog + testbench (+ bitstream, if --synthesize)")
    vlog.add_argument("--module-name", default="model", help="top-level Verilog module name (default: 'model')")
    vlog.add_argument("--samples", type=int, default=20, help="number of Icarus Verilog simulation samples to check (default: 20)")
    vlog.add_argument("--seed", type=int, default=0, help="RNG seed for simulation sample generation")
    vlog.add_argument("--synthesize", action="store_true", help="also attempt real synthesis via Yosys + NextPNR-ECP5 + ecppack (needs that toolchain on PATH); without this flag, only the Verilog + host simulation verification run")
    vlog.add_argument("--device", default="45k", choices=["12k", "25k", "45k", "85k", "um-25k", "um-45k", "um-85k", "um5g-25k", "um5g-45k", "um5g-85k"], help="ECP5 device size for --synthesize (default: 45k, i.e. LFE5U-45F)")
    vlog.add_argument("--package", default="CABGA381", help="device package for --synthesize (default: CABGA381)")
    vlog.add_argument("--freq", type=float, default=12.0, help="target clock frequency in MHz for --synthesize's timing check (default: 12)")

    return p


def _load_board(board_id: str, boards_dir: Path | None):
    """Returns the board, or None (after printing the error) on failure --
    never raises/exits directly, so `main()` stays a plain function callers
    can invoke without triggering an actual process exit (tests call it this way)."""
    try:
        return BoardRegistry(boards_dir).get(board_id)
    except EdgeForgeError as e:
        print(f"error: {e}", file=sys.stderr)
        return None


def cmd_list_boards(args: argparse.Namespace) -> int:
    reg = BoardRegistry(args.boards_dir)
    try:
        boards = sorted(reg.all().values(), key=lambda b: b.id)
    except EdgeForgeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not boards:
        print(f"no boards found in {reg.boards_dir}")
        return 0
    print(f"boards in {reg.boards_dir}:")
    for b in boards:
        print(f"  {b.id:20s} {b.name}")
        print(f"  {'':20s} tier={b.model_tier:10s} arch={b.architecture:16s} flash={b.memory.flash_bytes}B ram={b.memory.ram_bytes}B compiler={b.toolchain.compiler}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    try:
        result = detect_and_ingest(args.model, sample_range=_parse_range(args.sample_range), task=args.task)
    except EdgeForgeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(result.ir.describe())
    if args.board:
        board = _load_board(args.board, args.boards_dir)
        if board is None:
            return 1
        try:
            report = check_budget(result.ir, board)
        except EdgeForgeError as e:
            print(f"\nerror: {e}", file=sys.stderr)
            return 1
        print()
        print(report.describe(board))
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    board = _load_board(args.board, args.boards_dir)
    if board is None:
        return 1

    print(f"== converting {args.model} for '{board.id}' ==")
    result = run_conversion(
        args.model,
        board,
        args.out,
        model_name=args.model_name,
        task=args.task,
        sample_range=_parse_range(args.sample_range),
        representative_data=_load_rep_data(args.rep_data),
        do_build=not args.no_build,
        do_validate=not args.no_validate,
        samples=args.samples,
        seed=args.seed,
    )

    if result.ingest_result:
        print(result.ingest_result.ir.describe())
    if result.error and result.error_stage == "ingest":
        print(f"error: {result.error}", file=sys.stderr)
        return 1

    if result.footprint:
        print(f"\n== footprint against board '{board.id}' ==")
        print(result.footprint.describe(board))
    if result.error and result.error_stage == "footprint":
        print(f"error: {result.error}", file=sys.stderr)
        return 1

    if result.generated:
        names = [p.name for p in [result.generated.model_h, result.generated.model_c, result.generated.main_c, result.generated.linker_script, result.generated.startup_c, result.generated.sketch_ino] if p]
        print(f"\n== generated C source into {args.out} ==")
        print("generated: " + ", ".join(names))

    if result.build:
        print(f"\n== build for '{board.id}' ==")
        print(result.build.summary())

    if result.golden:
        print(f"\n== golden-vector validation ({args.samples} samples) ==")
        print(result.golden.describe())
    elif result.validate_skipped_reason:
        print("\n== golden-vector validation: skipped ==")
        print(result.validate_skipped_reason)
    if result.error and result.error_stage == "validate":
        print(f"error: {result.error}", file=sys.stderr)
        return 1

    print()
    if result.ok:
        note = " (source generated; install a host C compiler to also verify it -- see the message above)" if result.validate_skipped_reason else ""
        print(f"convert: SUCCESS{note}")
        return 0
    build_bad = result.build and not result.build.success and not result.build.toolchain_missing
    validate_bad = result.golden and not result.golden.all_passed
    print(
        "convert: FAILED" + (" (build error)" if build_bad else "") + (" (golden-vector mismatch)" if validate_bad else ""),
        file=sys.stderr,
    )
    return 1


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from edgeforge.webui import create_app
    except ImportError as e:
        print(
            f"error: the web UI requires Flask, which is not installed ({e}). Install it with: pip install flask",
            file=sys.stderr,
        )
        return 1
    app = create_app(boards_dir=args.boards_dir)
    print(f"EdgeForge web UI: http://{args.host}:{args.port}  (local machine only unless --host is changed; Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def cmd_convert_fpga(args: argparse.Namespace) -> int:
    try:
        from edgeforge.fpga.convert import HlsToolNotFoundError, convert_and_validate
    except ImportError as e:
        print(
            f"error: the FPGA backend requires the 'hls4ml' package, which failed to import ({e}). "
            "Install it with: pip install hls4ml",
            file=sys.stderr,
        )
        return 1

    print(f"== converting {args.model} for FPGA part '{args.part}' (backend={args.backend}) ==")
    try:
        result = convert_and_validate(
            args.model,
            args.out,
            part=args.part,
            backend=args.backend,
            precision=args.precision,
            reuse_factor=args.reuse_factor,
            sample_range=_parse_range(args.sample_range) or (0.0, 1.0),
            n_samples=args.samples,
            seed=args.seed,
            tolerance=args.tolerance,
            attempt_synthesis=args.synthesize,
        )
    except HlsToolNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"model: {result.param_count} parameters")
    print(f"HLS project generated: {result.output_dir}")

    if result.validation is None:
        print(f"error: {result.error}", file=sys.stderr)
        return 1

    print(f"\n== HLS C-simulation validation ({args.samples} samples) ==")
    print(result.validation.describe())

    if args.synthesize:
        print("\n== Vivado HLS synthesis ==")
        if result.toolchain_missing:
            print(
                f"toolchain '{result.toolchain_missing}' not found on this machine ({result.install_hint}); "
                "HLS project generated and host-verified, but not synthesized"
            )
        elif result.synth_ok:
            print(f"synthesis OK -- report: {result.synth_report}")
        else:
            print(f"synthesis failed: {result.error}", file=sys.stderr)

    print()
    if result.ok:
        note = "" if not args.synthesize or result.synth_ok else " (source generated + host-verified; synthesis not completed, see above)"
        print(f"convert-fpga: SUCCESS{note}")
        return 0
    print("convert-fpga: FAILED (C-simulation diverged from the original model)", file=sys.stderr)
    return 1


def cmd_convert_verilog(args: argparse.Namespace) -> int:
    from edgeforge.verilog.convert import convert_and_validate

    print(f"== converting {args.model} to Verilog (module '{args.module_name}') ==")
    try:
        result = convert_and_validate(
            args.model,
            args.out,
            module_name=args.module_name,
            n_samples=args.samples,
            seed=args.seed,
            attempt_synthesis=args.synthesize,
            device=args.device,
            package=args.package,
            freq_mhz=args.freq,
        )
    except EdgeForgeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"model: {result.param_count} parameters")
    print(f"Verilog + testbench generated: {result.output_dir}")

    print(f"\n== Icarus Verilog simulation ({args.samples} samples) ==")
    print(result.simulation.describe())

    if args.synthesize:
        print(f"\n== ECP5 synthesis ({args.device}, {args.package}, {args.freq} MHz target) ==")
        if result.toolchain_missing:
            print(
                f"toolchain '{result.toolchain_missing}' not found on this machine ({result.install_hint}); "
                "Verilog generated and host-verified, but not synthesized"
            )
        elif result.synth_ok:
            print(f"synthesis OK --\n{result.resource_report}")
        else:
            print(f"synthesis failed: {result.error}", file=sys.stderr)

    print()
    if result.ok:
        note = "" if not args.synthesize or result.synth_ok else " (source generated + host-verified; synthesis not completed, see above)"
        print(f"convert-verilog: SUCCESS{note}")
        return 0
    print("convert-verilog: FAILED (Verilog simulation diverged from the original model)", file=sys.stderr)
    return 1


def _load_rep_data(path: Path | None):
    if path is None:
        return None
    import numpy as np

    return np.load(path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-boards":
        return cmd_list_boards(args)
    if args.command == "inspect":
        return cmd_inspect(args)
    if args.command == "convert":
        return cmd_convert(args)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "convert-fpga":
        return cmd_convert_fpga(args)
    if args.command == "convert-verilog":
        return cmd_convert_verilog(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
