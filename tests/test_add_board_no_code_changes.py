"""The core architectural claim of this project: adding a board means writing
a YAML file, not touching generator/build/validate code. This test defines a
brand-new board (never imported, never special-cased anywhere in edgeforge)
purely as a YAML file in a throwaway directory, and drives it through the
exact same registry -> ingest -> footprint -> codegen -> build -> validate
pipeline every shipped board uses, with zero edgeforge code changes.
"""

import shutil

import pytest
import yaml

from edgeforge.boards.registry import BoardRegistry
from edgeforge.build.toolchain import build_firmware
from edgeforge.codegen.generator import generate
from edgeforge.ingest import sklearn_ingest
from edgeforge.quantize.footprint import check_budget
from edgeforge.validate.golden import run_golden_validation

_NOVEL_BOARD_YAML = """
id: test_novel_board
name: "A board invented by this test, never referenced anywhere in edgeforge"
architecture: arm-cortex-m0
bits: 32
endianness: little
clock_hz: 48000000
model_tier: classical
memory:
  flash_bytes: 65536
  ram_bytes: 16384
  flash_reserved_bytes: 4096
  ram_reserved_bytes: 2048
c_dialect:
  types:
    weight_i8: int8_t
    bias_i32: int32_t
    activation_i8: int8_t
    quant_accum: int32_t
    real: float
    classical_accum: float
    index: uint8_t
    class_index: uint8_t
  rom_qualifier: ""
  rom_guard_macro: null
  ram_qualifier: ""
  ram_guard_macro: null
  function_attributes: ""
  extra_includes: []
toolchain:
  compiler: arm-none-eabi-gcc
  compile_args: ["-mcpu=cortex-m0", "-mthumb", "-mfloat-abi=soft", "-Os", "-std=c99", "-ffreestanding", "-Wall"]
  link_args: ["-nostdlib", "-Wl,--gc-sections", "-lgcc"]
  linker_script: "generic.ld.j2"
  linker_origins: {flash: 0x08000000, ram: 0x20000000}
  objcopy: arm-none-eabi-objcopy
  output_format: ihex
  needs_startup_stub: true
  install_hint: "apt-get install gcc-arm-none-eabi"
host_test:
  compiler: gcc
  compile_args: ["-std=c99", "-O2", "-Wall"]
  link_args: ["-lm"]
"""

requires_arm_gcc = pytest.mark.skipif(shutil.which("arm-none-eabi-gcc") is None, reason="arm-none-eabi-gcc not installed")


@pytest.fixture
def novel_boards_dir(tmp_path):
    (tmp_path / "test_novel_board.yaml").write_text(_NOVEL_BOARD_YAML)
    return tmp_path


def test_novel_board_is_discovered_and_loads(novel_boards_dir):
    reg = BoardRegistry(novel_boards_dir)
    assert reg.list_ids() == ["test_novel_board"]
    board = reg.get("test_novel_board")
    assert board.id == "test_novel_board"
    assert board.model_tier == "classical"


def test_novel_board_goes_through_full_pipeline(tree_clf_path, novel_boards_dir, tmp_path):
    reg = BoardRegistry(novel_boards_dir)
    board = reg.get("test_novel_board")

    result = sklearn_ingest.ingest(tree_clf_path)
    check_budget(result.ir, board)  # raises if it doesn't fit; must not raise here

    out_dir = tmp_path / "gen"
    gen = generate(result.ir, board, out_dir)
    assert gen.model_c.is_file()
    assert gen.linker_script.is_file()  # generic.ld.j2, same template file every ARM board uses

    report = run_golden_validation(result, board, gen.model_c, out_dir, n_samples=15)
    assert report.all_passed, report.describe()


@requires_arm_gcc
def test_novel_board_actually_compiles(tree_clf_path, novel_boards_dir, tmp_path):
    reg = BoardRegistry(novel_boards_dir)
    board = reg.get("test_novel_board")
    result = sklearn_ingest.ingest(tree_clf_path)
    gen = generate(result.ir, board, tmp_path)
    res = build_firmware(gen, board)
    assert res.success, res.log
