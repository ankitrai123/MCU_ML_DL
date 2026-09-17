import pytest

from edgeforge.boards.registry import BoardRegistry
from edgeforge.errors import BoardError


def test_discovers_all_shipped_boards(boards_dir):
    reg = BoardRegistry(boards_dir)
    ids = set(reg.list_ids())
    assert {"stm32f411", "8051_at89s52", "native_cortex_m4", "esp32_native"} <= ids


def test_board_fields_are_typed_correctly(boards_dir):
    reg = BoardRegistry(boards_dir)
    stm32 = reg.get("stm32f411")
    assert stm32.model_tier == "deep"
    assert stm32.memory.flash_bytes == 524288
    assert stm32.toolchain.compiler == "arm-none-eabi-gcc"
    assert stm32.accepts_tier("classical")
    assert stm32.accepts_tier("deep")

    at89 = reg.get("8051_at89s52")
    assert at89.model_tier == "classical"
    assert at89.accepts_tier("classical")
    assert not at89.accepts_tier("deep")
    assert at89.c_dialect.rom_qualifier == "__code"
    assert at89.c_dialect.rom_guard_macro == "__SDCC"


def test_unknown_board_raises_with_available_list(boards_dir):
    reg = BoardRegistry(boards_dir)
    with pytest.raises(BoardError, match="unknown board id"):
        reg.get("not_a_real_board")


@pytest.mark.parametrize(
    "missing_field",
    ["id", "bits", "clock_hz", "model_tier", "memory", "c_dialect", "toolchain"],
)
def test_missing_required_field_raises_board_error(tmp_path, missing_field):
    raw = {
        "id": "broken",
        "bits": 32,
        "clock_hz": 100000000,
        "model_tier": "classical",
        "memory": {"flash_bytes": 1024, "ram_bytes": 256},
        "c_dialect": {
            "types": {
                "weight_i8": "int8_t", "bias_i32": "int32_t", "activation_i8": "int8_t",
                "quant_accum": "int32_t", "real": "float", "classical_accum": "float",
                "index": "uint16_t", "class_index": "uint8_t",
            }
        },
        "toolchain": {"compiler": "gcc"},
    }
    raw.pop(missing_field, None)
    board_file = tmp_path / "broken.yaml"
    import yaml

    board_file.write_text(yaml.safe_dump(raw))
    reg = BoardRegistry(tmp_path)
    with pytest.raises(BoardError):
        reg.get("broken")


def test_id_must_match_filename(tmp_path):
    import yaml

    raw = {"id": "wrong_id", "bits": 32, "clock_hz": 1, "model_tier": "classical",
           "memory": {"flash_bytes": 1, "ram_bytes": 1}, "c_dialect": {"types": {}}, "toolchain": {"compiler": "gcc"}}
    (tmp_path / "actual_name.yaml").write_text(yaml.safe_dump(raw))
    reg = BoardRegistry(tmp_path)
    with pytest.raises(BoardError, match="must match the filename stem"):
        reg.get("actual_name")
