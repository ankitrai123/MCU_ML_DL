from io import BytesIO

import pytest

flask = pytest.importorskip("flask")

from edgeforge.webui import create_app


@pytest.fixture
def client(boards_dir, tmp_path):
    app = create_app(boards_dir=boards_dir, runs_dir=tmp_path / "runs")
    app.config["TESTING"] = True
    return app.test_client()


def test_index_lists_boards(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"stm32f411" in r.data
    assert b"8051_at89s52" in r.data


def test_convert_tree_on_8051_succeeds(client, tree_clf_path):
    with open(tree_clf_path, "rb") as f:
        data = f.read()
    r = client.post(
        "/convert",
        data={"model_file": (BytesIO(data), "iris_tree.pkl"), "board_id": "8051_at89s52", "samples": "10"},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    text = r.data.decode()
    assert "success" in text.lower()
    assert "matched" in text.lower()
    assert "generated/model.c" in text


def test_convert_then_download_file(client, tree_clf_path):
    import re

    with open(tree_clf_path, "rb") as f:
        data = f.read()
    r = client.post(
        "/convert",
        data={"model_file": (BytesIO(data), "iris_tree.pkl"), "board_id": "stm32f411", "samples": "5"},
        content_type="multipart/form-data",
    )
    run_id = re.search(r"/runs/([a-f0-9]+)/download/", r.data.decode()).group(1)

    r2 = client.get(f"/runs/{run_id}/download/generated/model.c")
    assert r2.status_code == 200
    assert b"model_infer" in r2.data

    r3 = client.get(f"/runs/{run_id}/download-all")
    assert r3.status_code == 200
    assert r3.data[:2] == b"PK"  # zip magic bytes


def test_download_rejects_path_traversal(client, tree_clf_path):
    import re

    with open(tree_clf_path, "rb") as f:
        data = f.read()
    r = client.post(
        "/convert",
        data={"model_file": (BytesIO(data), "iris_tree.pkl"), "board_id": "stm32f411", "samples": "5"},
        content_type="multipart/form-data",
    )
    run_id = re.search(r"/runs/([a-f0-9]+)/download/", r.data.decode()).group(1)

    assert client.get("/runs/../../etc/download/passwd").status_code == 404
    assert client.get(f"/runs/{run_id}/download/../../../etc/passwd").status_code == 404
    assert client.get("/runs/not-a-real-run/download/model.c").status_code == 404


def test_unknown_board_flashes_error(client, tree_clf_path):
    with open(tree_clf_path, "rb") as f:
        data = f.read()
    r = client.post(
        "/convert",
        data={"model_file": (BytesIO(data), "iris_tree.pkl"), "board_id": "not_a_board", "samples": "5"},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert r.status_code == 200
    assert b"Board error" in r.data


def test_rejects_unrecognized_extension(client):
    r = client.post(
        "/convert",
        data={"model_file": (BytesIO(b"not a model"), "model.txt"), "board_id": "stm32f411"},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert r.status_code == 200
    assert b"unrecognized extension" in r.data


def test_no_file_flashes_error(client):
    r = client.post(
        "/convert",
        data={"board_id": "stm32f411"},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert r.status_code == 200
    assert b"choose a model file" in r.data
