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
        names = [p.name for p in [result.generated.model_h, result.generated.model_c, result.generated.main_c, result.generated.linker_script, result.generated.startup_c] if p]
        print(f"\n== generated C source into {args.out} ==")
        print("generated: " + ", ".join(names))

    if result.build:
        print(f"\n== build for '{board.id}' ==")
        print(result.build.summary())

    if result.golden:
        print(f"\n== golden-vector validation ({args.samples} samples) ==")
        print(result.golden.describe())
    if result.error and result.error_stage == "validate":
        print(f"error: {result.error}", file=sys.stderr)
        return 1

    print()
    if result.ok:
        print("convert: SUCCESS")
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
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
