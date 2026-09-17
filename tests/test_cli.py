from edgeforge.cli import main


def test_list_boards(capsys, boards_dir):
    rc = main(["--boards-dir", str(boards_dir), "list-boards"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "stm32f411" in out
    assert "8051_at89s52" in out


def test_convert_tree_on_stm32_succeeds(tree_clf_path, boards_dir, tmp_path, capsys):
    rc = main([
        "--boards-dir", str(boards_dir),
        "convert", "--model", str(tree_clf_path), "--board", "stm32f411", "--out", str(tmp_path),
    ])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "convert: SUCCESS" in out
    assert (tmp_path / "model.c").is_file()


def test_convert_unknown_board_fails_cleanly(tree_clf_path, boards_dir, tmp_path):
    rc = main([
        "--boards-dir", str(boards_dir),
        "convert", "--model", str(tree_clf_path), "--board", "not_a_board", "--out", str(tmp_path),
    ])
    assert rc == 1


def test_convert_tier_mismatch_fails_cleanly(keras_mlp_path, boards_dir, tmp_path, capsys):
    rc = main([
        "--boards-dir", str(boards_dir),
        "convert", "--model", str(keras_mlp_path), "--board", "8051_at89s52", "--out", str(tmp_path),
        "--sample-range=-2,8",  # note the '=': argparse would otherwise mistake a leading '-' value for another flag
    ])
    assert rc == 1


def test_inspect_without_board(tree_clf_path, boards_dir, capsys):
    rc = main(["--boards-dir", str(boards_dir), "inspect", "--model", str(tree_clf_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "kind=classical" in out


def test_inspect_with_board_reports_footprint(tree_clf_path, boards_dir, capsys):
    rc = main(["--boards-dir", str(boards_dir), "inspect", "--model", str(tree_clf_path), "--board", "8051_at89s52"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "flash (weights)" in out
