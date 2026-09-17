"""A basic local web UI for EdgeForge: upload a model, pick a board, convert.

This is a convenience layer over `edgeforge.pipeline.run_conversion` for
end users who'd rather not use the CLI -- it does not reimplement any
ingest/codegen/build/validate logic. It's meant to run on localhost for a
single local user, the same way the CLI is: there's no auth, no multi-tenant
isolation, and -- since ingesting a .pkl means unpickling it -- no defense
against a malicious model file beyond "only convert models you trust,"
exactly the same caveat the CLI and README already carry. Don't bind this to
a public interface without addressing that.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from edgeforge.boards.registry import BoardRegistry
from edgeforge.errors import EdgeForgeError
from edgeforge.pipeline import run_conversion

ALLOWED_EXTENSIONS = (".pkl", ".pickle", ".h5", ".keras", ".onnx")
MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # 64MB: generous for a small model, not for abuse
RUN_MAX_AGE_SECONDS = 3600  # best-effort cleanup of old runs on each request, see _cleanup_old_runs


def _parse_range(s: str) -> Optional[tuple[float, float]]:
    s = (s or "").strip()
    if not s:
        return None
    lo, hi = s.split(",")
    return float(lo), float(hi)


def _cleanup_old_runs(runs_dir: Path) -> None:
    now = time.time()
    for child in runs_dir.iterdir():
        try:
            if child.is_dir() and (now - child.stat().st_mtime) > RUN_MAX_AGE_SECONDS:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            pass  # best-effort; a run directory racing with cleanup is not worth failing the request over


def _safe_run_dir(runs_dir: Path, run_id: str) -> Path:
    """Resolves run_id to a directory under runs_dir, rejecting any path-traversal attempt."""
    if not run_id or any(c in run_id for c in ("/", "\\", "..")):
        abort(404)
    candidate = (runs_dir / run_id).resolve()
    if candidate.parent != runs_dir.resolve() or not candidate.is_dir():
        abort(404)
    return candidate


def create_app(boards_dir: Optional[Path] = None, runs_dir: Optional[Path] = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["EDGEFORGE_BOARDS_DIR"] = boards_dir
    app.config["EDGEFORGE_RUNS_DIR"] = Path(runs_dir) if runs_dir else Path(tempfile.gettempdir()) / "edgeforge_webui_runs"
    app.config["EDGEFORGE_RUNS_DIR"].mkdir(parents=True, exist_ok=True)
    app.secret_key = "edgeforge-local-webui"  # local single-user tool: session data never leaves this machine

    def _boards():
        return sorted(BoardRegistry(app.config["EDGEFORGE_BOARDS_DIR"]).all().values(), key=lambda b: b.id)

    @app.route("/")
    def index():
        return render_template("index.html", boards=_boards())

    @app.route("/convert", methods=["POST"])
    def convert():
        runs_dir = app.config["EDGEFORGE_RUNS_DIR"]
        _cleanup_old_runs(runs_dir)

        upload = request.files.get("model_file")
        if not upload or not upload.filename:
            flash("Please choose a model file to upload.")
            return redirect(url_for("index"))

        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            flash(f"'{upload.filename}' has an unrecognized extension. Expected one of: {', '.join(ALLOWED_EXTENSIONS)}")
            return redirect(url_for("index"))

        board_id = request.form.get("board_id", "")
        try:
            board = BoardRegistry(app.config["EDGEFORGE_BOARDS_DIR"]).get(board_id)
        except EdgeForgeError as e:
            flash(f"Board error: {e}")
            return redirect(url_for("index"))

        run_id = uuid.uuid4().hex
        run_dir = runs_dir / run_id
        run_dir.mkdir()
        model_path = run_dir / secure_filename(upload.filename)
        upload.save(model_path)

        try:
            sample_range = _parse_range(request.form.get("sample_range", ""))
        except ValueError:
            flash("Sample range must look like 'lo,hi', e.g. -2,8")
            return redirect(url_for("index"))

        task = request.form.get("task", "auto")
        samples = request.form.get("samples", "20")
        samples = int(samples) if samples.strip().isdigit() else 20

        out_dir = run_dir / "generated"
        try:
            result = run_conversion(
                model_path,
                board,
                out_dir,
                task=task,
                sample_range=sample_range,
                samples=samples,
            )
        except Exception as e:  # a genuinely unexpected failure, not an EdgeForgeError run_conversion already caught
            return render_template("result.html", run_id=run_id, board=board, crash=str(e))

        files = []
        if out_dir.is_dir():
            files = sorted(p.relative_to(run_dir).as_posix() for p in out_dir.rglob("*") if p.is_file())

        return render_template("result.html", run_id=run_id, board=board, result=result, files=files)

    @app.route("/runs/<run_id>/download/<path:filename>")
    def download_file(run_id, filename):
        run_dir = _safe_run_dir(app.config["EDGEFORGE_RUNS_DIR"], run_id)
        target = (run_dir / filename).resolve()
        if run_dir not in target.parents or not target.is_file():
            abort(404)
        return send_file(target, as_attachment=True)

    @app.route("/runs/<run_id>/download-all")
    def download_all(run_id):
        run_dir = _safe_run_dir(app.config["EDGEFORGE_RUNS_DIR"], run_id)
        zip_path = run_dir / "edgeforge_output.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in (run_dir / "generated").rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(run_dir / "generated"))
        return send_file(zip_path, as_attachment=True, download_name="edgeforge_output.zip")

    return app
