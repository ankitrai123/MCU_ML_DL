import re
from io import BytesIO
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")

from edgeforge.webui import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(boards_dir, tmp_path):
    app = create_app(boards_dir=boards_dir, runs_dir=tmp_path / "runs")
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture(scope="session")
def room_comfort_csv() -> Path:
    return REPO_ROOT / "examples" / "room_comfort.csv"


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


def _upload_csv(client, csv_path):
    with open(csv_path, "rb") as f:
        data = f.read()
    r = client.post(
        "/train",
        data={"csv_file": (BytesIO(data), csv_path.name)},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.data.decode()
    run_id = re.search(r'action="/train/([a-f0-9]+)/run"', r.data.decode()).group(1)
    return run_id


def test_train_form_loads(client):
    r = client.get("/train")
    assert r.status_code == 200
    assert b"Train a model" in r.data


def test_train_upload_shows_columns(client, room_comfort_csv):
    run_id = _upload_csv(client, room_comfort_csv)
    assert run_id


def test_train_upload_rejects_non_csv(client):
    r = client.post(
        "/train",
        data={"csv_file": (BytesIO(b"not a csv"), "data.txt")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert r.status_code == 200
    assert b"doesn" in r.data  # "doesn't look like a CSV file"


def test_train_upload_no_file_flashes_error(client):
    r = client.post("/train", data={}, content_type="multipart/form-data", follow_redirects=True)
    assert r.status_code == 200
    assert b"choose a CSV file" in r.data


def test_train_run_logistic_regression_succeeds(client, room_comfort_csv):
    run_id = _upload_csv(client, room_comfort_csv)
    r = client.post(
        f"/train/{run_id}/run",
        data={"label_column": "comfort", "model_type": "logistic_regression", "test_size": "0.2", "seed": "0"},
    )
    assert r.status_code == 200
    text = r.data.decode()
    assert "accuracy" in text.lower()
    assert "Download model.pkl" in text


def test_train_run_bad_label_column_reshows_configure_page(client, room_comfort_csv):
    run_id = _upload_csv(client, room_comfort_csv)
    r = client.post(f"/train/{run_id}/run", data={"label_column": "not_a_column", "model_type": "logistic_regression"})
    assert r.status_code == 200
    assert b"not found in the CSV" in r.data
    assert b"temperature_c" in r.data  # configure page re-rendered with the same columns


def test_train_run_unknown_run_id_404s(client):
    r = client.post("/train/not-a-real-run/run", data={"label_column": "x", "model_type": "logistic_regression"})
    assert r.status_code == 404


def test_train_then_download_model(client, room_comfort_csv):
    run_id = _upload_csv(client, room_comfort_csv)
    client.post(f"/train/{run_id}/run", data={"label_column": "comfort", "model_type": "decision_tree_classifier"})
    r = client.get(f"/runs/{run_id}/download/model.pkl")
    assert r.status_code == 200
    assert len(r.data) > 0


def test_train_then_convert_chains_into_conversion(client, room_comfort_csv):
    run_id = _upload_csv(client, room_comfort_csv)
    client.post(f"/train/{run_id}/run", data={"label_column": "comfort", "model_type": "logistic_regression"})
    r = client.post(f"/train/{run_id}/convert", data={"board_id": "stm32f411"})
    assert r.status_code == 200
    text = r.data.decode()
    assert "success" in text.lower()
    assert "matched" in text.lower()


def test_convert_before_training_404s(client):
    # No prior /train/<run_id>/run call, so no model.pkl exists in a fresh run dir.
    run_id = "0" * 32
    r = client.post(f"/train/{run_id}/convert", data={"board_id": "stm32f411"})
    assert r.status_code == 404
