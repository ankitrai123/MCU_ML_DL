"""Invokes a board's declared toolchain (best-effort) and the always-available
host compiler (for golden-vector validation).

If the board's cross-compiler isn't installed, `build_firmware` returns a
result naming exactly what's missing and how to install it (each board's
`install_hint`) instead of raising -- the CLI can then still report the
generated source as a success and simply skip the compiled-artifact step, per
the spec's "if not, emit the source plus a clear message naming the
toolchain that's missing."
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from edgeforge.boards.schema import BoardProfile, ToolchainProfile
from edgeforge.codegen.generator import GeneratedFiles
from edgeforge.errors import BuildError

_OUTPUT_EXT = {"ihex": "hex", "binary": "bin"}


@dataclass
class SizeReport:
    flash_bytes: int
    ram_bytes: int


@dataclass
class BuildResult:
    success: bool
    artifact_path: Optional[Path] = None
    elf_path: Optional[Path] = None
    log: str = ""
    toolchain_missing: Optional[str] = None
    install_hint: str = ""
    size_report: Optional[SizeReport] = None

    def summary(self) -> str:
        if self.toolchain_missing:
            hint = f" ({self.install_hint})" if self.install_hint else ""
            return f"toolchain '{self.toolchain_missing}' not found on this machine{hint}; source was generated but not compiled"
        if not self.success:
            hint = ""
            if "consecutive bytes in internal RAM" in self.log or "ASlink-Error" in self.log:
                hint = (
                    "\n(this usually means the model's actual RAM use -- including SDCC's "
                    "non-reentrant per-function local-variable storage, which EdgeForge's "
                    "pre-flight estimate approximates but can't predict exactly -- exceeded "
                    "the board's RAM despite passing that estimate; try a smaller model)"
                )
            return f"build failed:\n{self.log}{hint}"
        parts = [f"build OK -> {self.artifact_path}"]
        if self.size_report:
            parts.append(f"  actual size: flash={self.size_report.flash_bytes}B ram={self.size_report.ram_bytes}B")
        return "\n".join(parts)


def _run(cmd: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


def _log(cmd: list[str], result: subprocess.CompletedProcess) -> str:
    return f"$ {' '.join(cmd)}\n{result.stdout}{result.stderr}"


def build_firmware(generated: GeneratedFiles, board: BoardProfile) -> BuildResult:
    tc = board.toolchain
    if shutil.which(tc.compiler) is None:
        return BuildResult(success=False, toolchain_missing=tc.compiler, install_hint=tc.install_hint)

    out_dir = generated.out_dir
    sources = generated.sources()
    log_parts: list[str] = []

    if tc.compile_per_file:
        objects = []
        for src in sources:
            obj = out_dir / f"{src.stem}.{tc.object_ext}"
            cmd = [tc.compiler, *tc.compile_args, "-c", str(src), "-o", str(obj)]
            result = _run(cmd, out_dir)
            log_parts.append(_log(cmd, result))
            if result.returncode != 0:
                return BuildResult(success=False, log="\n".join(log_parts))
            objects.append(obj)

        artifact_path = out_dir / f"firmware.{_ext_for(tc)}"
        cmd = [tc.compiler, *tc.compile_args, *[str(o) for o in objects], *tc.link_args, "-o", str(artifact_path)]
        result = _run(cmd, out_dir)
        log_parts.append(_log(cmd, result))
        if result.returncode != 0:
            return BuildResult(success=False, log="\n".join(log_parts))
        size_report = _sdcc_mem_size(out_dir / "firmware.mem")
        return BuildResult(success=True, artifact_path=artifact_path, elf_path=None, log="\n".join(log_parts), size_report=size_report)

    if tc.objcopy:
        elf_path = out_dir / "firmware.elf"
        cmd = [tc.compiler, *tc.compile_args]
        if tc.linker_script and generated.linker_script:
            cmd += ["-T", str(generated.linker_script)]
        # Libraries in link_args (e.g. -lgcc) must follow the object files that reference them --
        # a linker resolves archive symbols left-to-right and won't back-reference an earlier -l.
        cmd += [str(s) for s in sources] + list(tc.link_args) + ["-o", str(elf_path)]
        result = _run(cmd, out_dir)
        log_parts.append(_log(cmd, result))
        if result.returncode != 0:
            return BuildResult(success=False, log="\n".join(log_parts))

        artifact_path = out_dir / f"firmware.{_ext_for(tc)}"
        objcopy_fmt = tc.output_format if tc.output_format != "binary" else "binary"
        cmd2 = [tc.objcopy, "-O", objcopy_fmt, str(elf_path), str(artifact_path)]
        result2 = _run(cmd2, out_dir)
        log_parts.append(_log(cmd2, result2))
        if result2.returncode != 0:
            return BuildResult(success=False, log="\n".join(log_parts))

        size_report = _gnu_size(elf_path, tc)
        return BuildResult(success=True, artifact_path=artifact_path, elf_path=elf_path, log="\n".join(log_parts), size_report=size_report)

    artifact_path = out_dir / f"firmware.{_ext_for(tc)}"
    cmd = [tc.compiler, *tc.compile_args, *[str(s) for s in sources], *tc.link_args, "-o", str(artifact_path)]
    result = _run(cmd, out_dir)
    log_parts.append(_log(cmd, result))
    if result.returncode != 0:
        return BuildResult(success=False, log="\n".join(log_parts))
    return BuildResult(success=True, artifact_path=artifact_path, elf_path=None, log="\n".join(log_parts))


def build_host_executable(sources: list[Path], out_dir: Path, board: BoardProfile, exe_name: str = "a.out") -> BuildResult:
    """Compile `sources` with the host toolchain (always plain gcc, regardless of the target
    board's own compiler) -- used for golden-vector validation, which always runs on the host."""
    ht = board.host_test
    if shutil.which(ht.compiler) is None:
        return BuildResult(success=False, toolchain_missing=ht.compiler, install_hint="install a host C compiler (e.g. gcc)")

    exe_path = out_dir / exe_name
    cmd = [ht.compiler, *ht.compile_args, *[str(s) for s in sources], *ht.link_args, "-o", str(exe_path)]
    result = _run(cmd, out_dir)
    log = _log(cmd, result)
    if result.returncode != 0:
        return BuildResult(success=False, log=log)
    return BuildResult(success=True, artifact_path=exe_path, log=log)


def _ext_for(tc: ToolchainProfile) -> str:
    if tc.output_format not in _OUTPUT_EXT:
        raise BuildError(f"toolchain.output_format {tc.output_format!r} has no known output file extension")
    return _OUTPUT_EXT[tc.output_format]


def _gnu_size(elf_path: Path, tc: ToolchainProfile) -> Optional[SizeReport]:
    size_tool = tc.objcopy.replace("objcopy", "size") if tc.objcopy and "objcopy" in tc.objcopy else None
    if not size_tool or shutil.which(size_tool) is None:
        return None
    try:
        result = subprocess.run([size_tool, str(elf_path)], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        line = result.stdout.strip().splitlines()[1]
        text, data, bss = (int(x) for x in line.split()[:3])
        return SizeReport(flash_bytes=text + data, ram_bytes=data + bss)
    except Exception:
        return None


def _sdcc_mem_size(mem_path: Path) -> Optional[SizeReport]:
    if not mem_path.is_file():
        return None
    try:
        text = mem_path.read_text()
        flash_match = re.search(r"ROM/EPROM/FLASH\s+\S+\s+\S+\s+(\d+)", text)
        stack_match = re.search(r"Stack starts at:\s*0x([0-9A-Fa-f]+)", text)
        if not flash_match or not stack_match:
            return None
        return SizeReport(flash_bytes=int(flash_match.group(1)), ram_bytes=int(stack_match.group(1), 16))
    except Exception:
        return None
