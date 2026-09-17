"""Dispatches a model file to the right ingest backend by extension."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from edgeforge.errors import UnsupportedModelError
from edgeforge.ingest.base import IngestResult

_KNOWN_EXTENSIONS = (".pkl", ".pickle", ".h5", ".keras", ".onnx")


def detect_and_ingest(
    path: Path,
    sample_range: Optional[tuple[float, float]] = None,
    task: str = "auto",
    representative_data=None,
) -> IngestResult:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".pkl", ".pickle"):
        from edgeforge.ingest import sklearn_ingest

        return sklearn_ingest.ingest(path, sample_range=sample_range)
    if suffix in (".h5", ".keras"):
        from edgeforge.ingest import keras_ingest

        return keras_ingest.ingest(path, representative_data=representative_data, sample_range=sample_range, task=task)
    if suffix == ".onnx":
        from edgeforge.ingest import onnx_ingest

        return onnx_ingest.ingest(path, sample_range=sample_range, task=task)

    raise UnsupportedModelError(
        f"'{path}' has an unrecognized extension '{suffix}'. EdgeForge ingests: "
        f"{', '.join(_KNOWN_EXTENSIONS)} (scikit-learn pickle, Keras/TF, and ONNX respectively)."
    )
