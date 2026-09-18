"""Model -> synthesizable Verilog for the Lattice ECP5 family, via a from-scratch RTL
generator (Phase 3, second backend). Unlike edgeforge.fpga.convert (which wraps hls4ml for
Xilinx/Vivado HLS), there's no existing "HLS for the open-source Lattice flow" tool to wrap --
ECP5's own toolchain (Yosys + NextPNR + Project Trellis) takes synthesizable Verilog/VHDL
directly, so EdgeForge generates that RTL itself here (edgeforge.verilog.codegen), from the
*same* ModelIR every other path uses, but through entirely new templates and a new (fixed-point,
since small FPGAs have no float unit -- see fixedpoint.py) numeric representation. This is a
deliberately separate command from `convert`/`convert-fpga`: no boards/*.yaml, no --board.

Same two-stage shape as the hls4ml backend: always host-verify (here, Icarus Verilog
simulation against the real source model -- edgeforge.verilog.simulate), and separately,
best-effort real synthesis (Yosys -> NextPNR-ECP5 -> ecppack) gated on that toolchain being
installed. Real synthesis here targets the ECP5 device in general -- LUT/DSP counts and timing
are genuine, but with no board-specific pin-constraint (.lpf) file, every port lands on
whatever pin NextPNR finds free, which (for this module's wide flattened input/output buses)
uses nearly every I/O pin on a small part. That's fine for "does this fit and time-close",
not a drop-in bitstream for a specific board -- a real deployment supplies its own .lpf and
wires this module to on-chip logic (UART/SPI/BRAM) rather than driving chip pins directly.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from edgeforge.errors import EdgeForgeError
from edgeforge.ingest import detect_and_ingest
from edgeforge.verilog import codegen
from edgeforge.verilog.simulate import VerilogSimReport, run_verilog_simulation

DEFAULT_DEVICE = "45k"  # LFE5U-45F -- a common mid-range ECP5 part (e.g. on the ULX3S)
DEFAULT_PACKAGE = "CABGA381"
DEFAULT_FREQ_MHZ = 12
VALID_DEVICES = ("12k", "25k", "45k", "85k", "um-25k", "um-45k", "um-85k", "um5g-25k", "um5g-45k", "um5g-85k")


@dataclass
class VerilogBuildResult:
    ok: bool
    output_dir: Optional[Path] = None
    param_count: int = 0
    simulation: Optional[VerilogSimReport] = None
    synth_attempted: bool = False
    synth_ok: bool = False
    resource_report: str = ""
    toolchain_missing: Optional[str] = None
    install_hint: str = ""
    error: Optional[str] = None


def _run(cmd: list[str], cwd: Path, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


def convert_and_validate(
    model_path: Path,
    out_dir: Path,
    module_name: str = "model",
    n_samples: int = 20,
    seed: int = 0,
    attempt_synthesis: bool = False,
    device: str = DEFAULT_DEVICE,
    package: str = DEFAULT_PACKAGE,
    freq_mhz: float = DEFAULT_FREQ_MHZ,
) -> VerilogBuildResult:
    if device not in VALID_DEVICES:
        raise EdgeForgeError(f"device {device!r} is not a known ECP5 size -- one of: {', '.join(VALID_DEVICES)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ingest_result = detect_and_ingest(model_path)
    ir = ingest_result.ir
    param_count = ir.param_count()

    # validate_codegen_support runs again inside render_model, but running it first gives a
    # clean error before any simulation/synthesis work starts, matching cmd_convert's own
    # "fail at the earliest stage that can tell you something's wrong" ordering.
    codegen.validate_codegen_support(ir)

    simulation = run_verilog_simulation(ingest_result, out_dir, module_name=module_name, n_samples=n_samples, seed=seed)
    result = VerilogBuildResult(
        ok=simulation.all_passed, output_dir=out_dir, param_count=param_count, simulation=simulation
    )

    if attempt_synthesis:
        missing = [t for t in ("yosys", "nextpnr-ecp5", "ecppack") if shutil.which(t) is None]
        if missing:
            result.toolchain_missing = missing[0]
            result.install_hint = (
                "Install the open-source Lattice ECP5 toolchain: apt-get install yosys nextpnr-ecp5 "
                "fpga-trellis fpga-trellis-database (Debian/Ubuntu), or see "
                "https://github.com/YosysHQ/oss-cad-suite-build for a prebuilt cross-platform bundle."
            )
        else:
            result.synth_attempted = True
            try:
                result.resource_report, result.synth_ok = _synthesize(out_dir, module_name, device, package, freq_mhz)
            except _SynthesisFailed as e:
                result.error = str(e)

    return result


class _SynthesisFailed(EdgeForgeError):
    pass


def _synthesize(out_dir: Path, module_name: str, device: str, package: str, freq_mhz: float) -> tuple[str, bool]:
    model_v = out_dir / f"{module_name}.v"
    json_path = out_dir / f"{module_name}.json"
    config_path = out_dir / f"{module_name}.config"
    bit_path = out_dir / f"{module_name}.bit"

    yosys_cmd = ["yosys", "-p", f"read_verilog {model_v.name}; synth_ecp5 -top {module_name} -json {json_path.name}"]
    yosys_res = _run(yosys_cmd, out_dir)
    if yosys_res.returncode != 0:
        raise _SynthesisFailed(f"yosys synthesis failed:\n$ {' '.join(yosys_cmd)}\n{yosys_res.stdout}{yosys_res.stderr}")

    device_flag = f"--{device}"
    pnr_cmd = [
        "nextpnr-ecp5", "--json", json_path.name, "--textcfg", config_path.name,
        device_flag, "--package", package, "--freq", str(freq_mhz),
    ]
    pnr_res = _run(pnr_cmd, out_dir)
    # nextpnr writes its utilisation/timing report to stderr, not stdout.
    report = _extract_report(pnr_res.stderr)
    if pnr_res.returncode != 0:
        raise _SynthesisFailed(f"nextpnr-ecp5 place-and-route failed:\n$ {' '.join(pnr_cmd)}\n{report}")

    pack_cmd = ["ecppack", config_path.name, bit_path.name]
    pack_res = _run(pack_cmd, out_dir)
    if pack_res.returncode != 0:
        raise _SynthesisFailed(f"ecppack failed:\n$ {' '.join(pack_cmd)}\n{pack_res.stdout}{pack_res.stderr}")

    return report, bit_path.is_file()


def _extract_report(nextpnr_stderr: str) -> str:
    """The device-utilisation table and max-frequency line out of nextpnr's (chatty) log --
    resource/timing numbers a user actually wants, not the full placer/router trace."""
    lines = nextpnr_stderr.splitlines()
    keep = [ln for ln in lines if re.search(r"Device utilisation|Info: \s*\S+:\s*\d+/\s*\d+|Max frequency", ln)]
    return "\n".join(keep) if keep else nextpnr_stderr[-2000:]
