import shutil

import pytest

from edgeforge.boards.registry import BoardRegistry
from edgeforge.build.toolchain import build_firmware
from edgeforge.codegen.generator import generate
from edgeforge.ingest import sklearn_ingest

requires_sdcc = pytest.mark.skipif(shutil.which("sdcc") is None, reason="sdcc not installed")
requires_arm_gcc = pytest.mark.skipif(shutil.which("arm-none-eabi-gcc") is None, reason="arm-none-eabi-gcc not installed")


def test_generate_produces_expected_files(tree_clf_path, boards_dir, tmp_path):
    reg = BoardRegistry(boards_dir)
    result = sklearn_ingest.ingest(tree_clf_path)
    gen = generate(result.ir, reg.get("stm32f411"), tmp_path)
    assert gen.model_h.is_file()
    assert gen.model_c.is_file()
    assert gen.main_c.is_file()
    assert gen.linker_script.is_file()
    assert gen.startup_c.is_file()


def test_generate_8051_has_no_linker_script_or_startup(tree_clf_path, boards_dir, tmp_path):
    reg = BoardRegistry(boards_dir)
    result = sklearn_ingest.ingest(tree_clf_path)
    gen = generate(result.ir, reg.get("8051_at89s52"), tmp_path)
    assert gen.linker_script is None
    assert gen.startup_c is None


@requires_arm_gcc
def test_build_firmware_stm32(tree_clf_path, boards_dir, tmp_path):
    reg = BoardRegistry(boards_dir)
    result = sklearn_ingest.ingest(tree_clf_path)
    board = reg.get("stm32f411")
    gen = generate(result.ir, board, tmp_path)
    res = build_firmware(gen, board)
    assert res.success, res.log
    assert res.artifact_path.is_file()


@requires_sdcc
def test_build_firmware_8051(tree_clf_path, boards_dir, tmp_path):
    reg = BoardRegistry(boards_dir)
    result = sklearn_ingest.ingest(tree_clf_path)
    board = reg.get("8051_at89s52")
    gen = generate(result.ir, board, tmp_path)
    res = build_firmware(gen, board)
    assert res.success, res.log
    assert res.artifact_path.is_file()
    assert res.size_report is not None
    assert res.size_report.flash_bytes > 0


def test_missing_toolchain_reports_clearly(tree_clf_path, boards_dir, tmp_path, monkeypatch):
    reg = BoardRegistry(boards_dir)
    result = sklearn_ingest.ingest(tree_clf_path)
    board = reg.get("stm32f411")
    gen = generate(result.ir, board, tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: None)
    res = build_firmware(gen, board)
    assert not res.success
    assert res.toolchain_missing == "arm-none-eabi-gcc"
    assert "not found" in res.summary()
