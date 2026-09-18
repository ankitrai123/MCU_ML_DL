"""The Arduino Nano 33 BLE Sense Rev2 board: the first shipped board whose toolchain.kind
is "arduino-cli" instead of a direct cross-compiler invocation. Covers the sketch-folder
codegen shape and that golden-vector validation (host-only, independent of the target
toolchain) fully proves correctness even where arduino-cli itself isn't installed.
"""

import shutil

import pytest

from edgeforge.boards.registry import BoardRegistry
from edgeforge.build.toolchain import build_firmware
from edgeforge.codegen.generator import generate
from edgeforge.ingest import sklearn_ingest
from edgeforge.quantize.footprint import check_budget
from edgeforge.validate.golden import run_golden_validation

requires_arduino_cli = pytest.mark.skipif(shutil.which("arduino-cli") is None, reason="arduino-cli not installed")

BOARD_ID = "arduino_nano33_ble_sense_rev2"


def test_board_discovered_with_arduino_cli_toolchain(boards_dir):
    board = BoardRegistry(boards_dir).get(BOARD_ID)
    assert board.model_tier == "deep"
    assert board.toolchain.kind == "arduino-cli"
    assert board.toolchain.compiler == "arduino-cli"
    assert board.toolchain.fqbn == "arduino:mbed_nano:nano33ble"


def test_generate_produces_a_sketch_folder_not_main_c(tree_clf_path, boards_dir, tmp_path):
    board = BoardRegistry(boards_dir).get(BOARD_ID)
    result = sklearn_ingest.ingest(tree_clf_path)
    gen = generate(result.ir, board, tmp_path)

    assert gen.main_c is None
    assert gen.linker_script is None
    assert gen.startup_c is None
    assert gen.sketch_ino is not None
    assert gen.sketch_ino.is_file()
    assert gen.sketch_ino.name == "model_sketch.ino"
    # The .ino must share its containing directory's name -- an Arduino/arduino-cli requirement.
    assert gen.sketch_ino.parent.name == "model_sketch"
    # model.h/model.c live alongside the .ino (arduino-cli compiles every file in the sketch
    # directory together), not flat in out_dir like a raw-toolchain board.
    assert gen.model_h.parent == gen.sketch_ino.parent
    assert gen.model_c.parent == gen.sketch_ino.parent

    ino_text = gen.sketch_ino.read_text()
    assert "void setup(void)" in ino_text
    assert "void loop(void)" in ino_text
    assert "int main(" not in ino_text  # Arduino's own core supplies main(), not this sketch

    assert 'extern "C"' in gen.model_h.read_text()


def test_footprint_check_against_arduino_board(tree_clf_path, boards_dir):
    board = BoardRegistry(boards_dir).get(BOARD_ID)
    result = sklearn_ingest.ingest(tree_clf_path)
    report = check_budget(result.ir, board)  # must not raise
    assert report.param_count > 0


def test_golden_validation_passes_regardless_of_arduino_cli_availability(tree_clf_path, boards_dir, tmp_path):
    """Golden validation always compiles with the host compiler (never the target toolchain),
    so it fully proves correctness for this board whether or not arduino-cli happens to be
    installed on the machine running the test."""
    board = BoardRegistry(boards_dir).get(BOARD_ID)
    result = sklearn_ingest.ingest(tree_clf_path)
    gen = generate(result.ir, board, tmp_path)
    report = run_golden_validation(result, board, gen.model_c, tmp_path, n_samples=15)
    assert report.all_passed, report.describe()


def test_missing_arduino_cli_degrades_gracefully(tree_clf_path, boards_dir, tmp_path, monkeypatch):
    board = BoardRegistry(boards_dir).get(BOARD_ID)
    result = sklearn_ingest.ingest(tree_clf_path)
    gen = generate(result.ir, board, tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: None)
    res = build_firmware(gen, board)
    assert not res.success
    assert res.toolchain_missing == "arduino-cli"
    assert "arduino-cli" in res.install_hint
    assert "not found" in res.summary()


@requires_arduino_cli
def test_build_firmware_arduino_cli(tree_clf_path, boards_dir, tmp_path):
    board = BoardRegistry(boards_dir).get(BOARD_ID)
    result = sklearn_ingest.ingest(tree_clf_path)
    gen = generate(result.ir, board, tmp_path)
    res = build_firmware(gen, board)
    assert res.success, res.log
