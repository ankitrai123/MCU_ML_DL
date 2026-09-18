"""Board profile schema: the typed view of a board YAML file.

Every field the code generator, footprint checker, and toolchain runner need
is declared here and validated on load. Nothing downstream should ever branch
on ``board.id`` or ``board.architecture`` — those two fields exist purely for
human-readable reporting. Behavioral differences between boards must be
expressed as *data* on this object (a type string, a qualifier, a toolchain
command template), never as a Python ``if board.id == "...":``. If you find
yourself wanting to add such a branch, the fix is a new field here plus a
value for it in the board's YAML, not a special case in generator code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from edgeforge.errors import BoardError

# Logical type "roles" every board must map to a concrete C type string.
# Templates reference these by role (e.g. `types.quant_accum`), never by
# spelling out a C type literally, so a board can widen/narrow any of them
# without touching generator code.
REQUIRED_TYPE_ROLES = (
    "weight_i8",  # quantized int8 weight elements
    "bias_i32",  # quantized int32 bias elements
    "activation_i8",  # quantized int8 activation elements
    "quant_accum",  # accumulator for int8 x int8 dot products (deep tier)
    "real",  # floating point type used by the classical (unquantized) path
    "classical_accum",  # accumulator for classical dot products
    "index",  # loop counter / array index type
    "class_index",  # type used to hold an argmax class index
)

VALID_MODEL_TIERS = ("classical", "deep")
VALID_OUTPUT_FORMATS = ("ihex", "binary", "none")
# "raw": a direct cross-compiler invocation (compiler/compile_args/link_args/linker_script/
# objcopy all apply) -- every board before the Arduino one used this and nothing else.
# "arduino-cli": the board's own Arduino core owns the toolchain/startup/linking; EdgeForge
# only drives `arduino-cli compile --fqbn ...` against a generated sketch directory, so
# compile_args/link_args/linker_script/objcopy/needs_startup_stub go unused for this kind.
VALID_TOOLCHAIN_KINDS = ("raw", "arduino-cli")


@dataclass(frozen=True)
class MemoryBudget:
    flash_bytes: int
    ram_bytes: int
    flash_reserved_bytes: int = 0
    ram_reserved_bytes: int = 0
    # A non-reentrant 8-bit toolchain (SDCC's default calling convention) gives every function's
    # locals a permanent static slot rather than reusing stack space, so real RAM use runs well
    # above a simple sum of buffer sizes -- this multiplier reflects that toolchain characteristic
    # in the pre-flight estimate. 1.0 (no adjustment) is correct for a normal stack-based target.
    ram_overhead_multiplier: float = 1.0

    @property
    def available_flash_bytes(self) -> int:
        return self.flash_bytes - self.flash_reserved_bytes

    @property
    def available_ram_bytes(self) -> int:
        return self.ram_bytes - self.ram_reserved_bytes


@dataclass(frozen=True)
class CDialect:
    types: dict[str, str]
    rom_qualifier: str = "const"
    rom_guard_macro: Optional[str] = None
    ram_qualifier: str = ""
    ram_guard_macro: Optional[str] = None
    function_attributes: str = ""
    extra_includes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolchainProfile:
    compiler: str
    kind: str = "raw"  # "raw" (direct cross-compiler) or "arduino-cli" -- see VALID_TOOLCHAIN_KINDS
    fqbn: Optional[str] = None  # required when kind == "arduino-cli": the Arduino board id to compile against
    compile_args: tuple[str, ...] = ()
    link_args: tuple[str, ...] = ()
    linker_script: Optional[str] = None
    linker_origins: dict[str, int] = field(default_factory=dict)
    objcopy: Optional[str] = None
    output_format: str = "none"
    needs_startup_stub: bool = False
    compile_per_file: bool = False  # True: compiler can only take one source file per invocation
                                     # (SDCC); each source is compiled to an object file separately,
                                     # then all object files are linked in one further invocation.
    object_ext: str = "o"  # object file extension when compile_per_file is True (SDCC: "rel")
    install_hint: str = ""


@dataclass(frozen=True)
class HostTestProfile:
    compiler: str = "gcc"
    compile_args: tuple[str, ...] = ("-std=c99", "-O2", "-Wall")
    link_args: tuple[str, ...] = ("-lm",)


@dataclass(frozen=True)
class BoardProfile:
    id: str
    name: str
    architecture: str
    bits: int
    endianness: str
    clock_hz: int
    model_tier: str
    memory: MemoryBudget
    c_dialect: CDialect
    toolchain: ToolchainProfile
    host_test: HostTestProfile
    source_path: Path

    def type_of(self, role: str) -> str:
        try:
            return self.c_dialect.types[role]
        except KeyError:
            raise BoardError(
                f"board '{self.id}': c_dialect.types is missing required role '{role}' "
                f"(required roles: {', '.join(REQUIRED_TYPE_ROLES)})"
            ) from None

    def accepts_tier(self, model_kind_tier: str) -> bool:
        """Whether a model whose tier is `model_kind_tier` ('classical' or 'deep') may target this board."""
        if self.model_tier == "deep":
            return True  # deep boards have the resources to also run classical models
        return model_kind_tier == "classical"


def _require(d: dict, key: str, ctx: str) -> Any:
    if key not in d:
        raise BoardError(f"{ctx}: missing required field '{key}'")
    return d[key]


def _require_dict(d: dict, key: str, ctx: str) -> dict:
    v = _require(d, key, ctx)
    if not isinstance(v, dict):
        raise BoardError(f"{ctx}: field '{key}' must be a mapping, got {type(v).__name__}")
    return v


def _require_positive_int(d: dict, key: str, ctx: str) -> int:
    v = _require(d, key, ctx)
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
        raise BoardError(f"{ctx}: field '{key}' must be a positive integer, got {v!r}")
    return v


def parse_board_profile(raw: dict, source_path: Path) -> BoardProfile:
    """Validate a raw YAML dict and build a BoardProfile, or raise BoardError."""
    ctx = f"board file '{source_path}'"
    if not isinstance(raw, dict):
        raise BoardError(f"{ctx}: top-level YAML must be a mapping")

    board_id = _require(raw, "id", ctx)
    if not isinstance(board_id, str) or not board_id:
        raise BoardError(f"{ctx}: 'id' must be a non-empty string")
    if board_id != source_path.stem:
        raise BoardError(
            f"{ctx}: 'id' ({board_id!r}) must match the filename stem ({source_path.stem!r}) "
            f"so board ids are discoverable by filename alone"
        )

    name = raw.get("name", board_id)
    architecture = raw.get("architecture", "unknown")
    bits = raw.get("bits")
    if bits not in (8, 16, 32, 64):
        raise BoardError(f"{ctx}: 'bits' must be one of 8/16/32/64, got {bits!r}")
    endianness = raw.get("endianness", "little")
    if endianness not in ("little", "big"):
        raise BoardError(f"{ctx}: 'endianness' must be 'little' or 'big', got {endianness!r}")
    clock_hz = _require_positive_int(raw, "clock_hz", ctx)

    model_tier = _require(raw, "model_tier", ctx)
    if model_tier not in VALID_MODEL_TIERS:
        raise BoardError(f"{ctx}: 'model_tier' must be one of {VALID_MODEL_TIERS}, got {model_tier!r}")

    mem_raw = _require_dict(raw, "memory", ctx)
    memory = MemoryBudget(
        flash_bytes=_require_positive_int(mem_raw, "flash_bytes", f"{ctx} memory"),
        ram_bytes=_require_positive_int(mem_raw, "ram_bytes", f"{ctx} memory"),
        flash_reserved_bytes=int(mem_raw.get("flash_reserved_bytes", 0)),
        ram_reserved_bytes=int(mem_raw.get("ram_reserved_bytes", 0)),
        ram_overhead_multiplier=float(mem_raw.get("ram_overhead_multiplier", 1.0)),
    )
    if memory.available_flash_bytes <= 0:
        raise BoardError(
            f"{ctx}: memory.flash_reserved_bytes ({memory.flash_reserved_bytes}) leaves zero "
            f"or negative usable flash out of {memory.flash_bytes}"
        )
    if memory.available_ram_bytes <= 0:
        raise BoardError(
            f"{ctx}: memory.ram_reserved_bytes ({memory.ram_reserved_bytes}) leaves zero "
            f"or negative usable RAM out of {memory.ram_bytes}"
        )

    dialect_raw = _require_dict(raw, "c_dialect", ctx)
    types_raw = _require_dict(dialect_raw, "types", f"{ctx} c_dialect")
    missing_roles = [r for r in REQUIRED_TYPE_ROLES if r not in types_raw]
    if missing_roles:
        raise BoardError(
            f"{ctx}: c_dialect.types is missing required role(s): {', '.join(missing_roles)}"
        )
    extra_includes = tuple(dialect_raw.get("extra_includes", []) or [])
    c_dialect = CDialect(
        types={k: str(v) for k, v in types_raw.items()},
        rom_qualifier=str(dialect_raw.get("rom_qualifier", "const")),
        rom_guard_macro=dialect_raw.get("rom_guard_macro"),
        ram_qualifier=str(dialect_raw.get("ram_qualifier", "")),
        ram_guard_macro=dialect_raw.get("ram_guard_macro"),
        function_attributes=str(dialect_raw.get("function_attributes", "")),
        extra_includes=extra_includes,
    )

    tc_raw = _require_dict(raw, "toolchain", ctx)
    output_format = tc_raw.get("output_format", "none")
    if output_format not in VALID_OUTPUT_FORMATS:
        raise BoardError(
            f"{ctx}: toolchain.output_format must be one of {VALID_OUTPUT_FORMATS}, got {output_format!r}"
        )
    kind = tc_raw.get("kind", "raw")
    if kind not in VALID_TOOLCHAIN_KINDS:
        raise BoardError(f"{ctx}: toolchain.kind must be one of {VALID_TOOLCHAIN_KINDS}, got {kind!r}")
    fqbn = tc_raw.get("fqbn")
    if kind == "arduino-cli" and not fqbn:
        raise BoardError(f"{ctx}: toolchain.fqbn is required when toolchain.kind is 'arduino-cli'")
    toolchain = ToolchainProfile(
        compiler=_require(tc_raw, "compiler", f"{ctx} toolchain"),
        kind=kind,
        fqbn=fqbn,
        compile_args=tuple(tc_raw.get("compile_args", []) or []),
        link_args=tuple(tc_raw.get("link_args", []) or []),
        linker_script=tc_raw.get("linker_script"),
        linker_origins={k: int(v) for k, v in (tc_raw.get("linker_origins") or {}).items()},
        objcopy=tc_raw.get("objcopy"),
        output_format=output_format,
        needs_startup_stub=bool(tc_raw.get("needs_startup_stub", False)),
        compile_per_file=bool(tc_raw.get("compile_per_file", False)),
        object_ext=str(tc_raw.get("object_ext", "o")),
        install_hint=str(tc_raw.get("install_hint", "")),
    )

    ht_raw = raw.get("host_test", {}) or {}
    host_test = HostTestProfile(
        compiler=ht_raw.get("compiler", "gcc"),
        compile_args=tuple(ht_raw.get("compile_args", ["-std=c99", "-O2", "-Wall"])),
        link_args=tuple(ht_raw.get("link_args", ["-lm"])),
    )

    return BoardProfile(
        id=board_id,
        name=str(name),
        architecture=str(architecture),
        bits=bits,
        endianness=endianness,
        clock_hz=clock_hz,
        model_tier=model_tier,
        memory=memory,
        c_dialect=c_dialect,
        toolchain=toolchain,
        host_test=host_test,
        source_path=source_path,
    )
