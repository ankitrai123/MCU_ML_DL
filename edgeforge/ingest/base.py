"""Shared ingest result type and sample-input helpers.

Every ingest backend (sklearn/keras/onnx) lowers its model into the same
``IngestResult``: a ``ModelIR`` for codegen, plus a ``reference_fn`` that
calls the *real* source-framework model (not a reimplementation of the IR) so
golden-vector validation actually catches IR-extraction bugs, not just C
arithmetic bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from edgeforge.ir import ModelIR

# Given a sample real-valued input vector (matching ir.input_spec.shape,
# flattened), returns (predicted_class_index_or_None, raw_output_array).
ReferenceFn = Callable[[np.ndarray], tuple[Optional[int], np.ndarray]]

# Given a numpy Generator and a count N, returns an (N, input_size) array of
# plausible real-valued inputs for golden-vector sampling.
InputSampler = Callable[[np.random.Generator, int], np.ndarray]


@dataclass
class IngestResult:
    ir: ModelIR
    reference_fn: ReferenceFn
    input_sampler: InputSampler
    source_kind: str  # 'sklearn' | 'keras' | 'onnx'


def uniform_range_sampler(low: np.ndarray, high: np.ndarray) -> InputSampler:
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)

    def _sample(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(low, high, size=(n, low.shape[0]))

    return _sample
