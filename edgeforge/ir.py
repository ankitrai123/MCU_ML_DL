"""Intermediate representation shared by every ingest backend and every board.

This is the seam that makes the "board registry, not per-chip branching"
architecture work: sklearn/.onnx/.h5 ingesters all lower their model into a
``ModelIR`` built from the same small op vocabulary, and codegen renders that
*same* IR through the *same* Jinja2 templates regardless of which board it
targets -- only the board profile (types, qualifiers, toolchain) changes.

Op vocabulary
-------------
``tree``        decision tree: parallel node arrays (feature/threshold/children/leaf value)
``linear``      y = W x + b  (a plain affine layer -- covers logistic regression,
                 one MLP layer, or a quantized TFLite FullyConnected layer; which
                 kernel codegen emits depends on the *data* -- the weight
                 tensor's dtype -- not on which board it is)
``conv2d``      2D convolution, optionally with a fused activation (TFLite style)
``depthwise_conv2d``  depthwise 2D convolution
``maxpool2d``   2D max pooling
``activation``  elementwise nonlinearity: relu | relu6 | sigmoid | tanh | identity
``affine``      elementwise y = x*scale + shift, one (scale, shift) pair per input
                 feature -- used to fold a fitted sklearn StandardScaler/MinMaxScaler
                 into the generated C, so a raw (unscaled) sensor reading is normalized
                 on-device exactly the way training data was, instead of requiring the
                 caller to replicate that math by hand

Reshape/flatten ops from the source graph are elided during ingest (the
consuming node is rewired to read the pre-reshape tensor directly) since a
flat row-major C array needs no reshaping at all.

Every model, classical or deep, lowers to a linear chain of these nodes
ending in one output tensor. ``ModelIR.task`` selects the final wrapper the
codegen emits: an argmax over the (dequantized) output for classification, or
a plain dequantized value for regression -- both are thin, generic wrappers
around the same node pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from edgeforge.errors import UnsupportedOpError

SUPPORTED_OPS = (
    "tree",
    "linear",
    "conv2d",
    "depthwise_conv2d",
    "maxpool2d",
    "activation",
    "affine",
)

# dtype -> C storage size in bytes, used for footprint estimation.
DTYPE_SIZES = {
    "int8": 1,
    "uint8": 1,
    "int32": 4,
    "uint32": 4,
    "float32": 4,
}

TENSOR_ROLES = ("input", "output", "weight", "bias", "activation")


def _prod(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


@dataclass
class TensorSpec:
    """One tensor in the IR graph: an input, output, weight/bias constant, or
    an intermediate activation that needs a scratch buffer in C."""

    name: str
    shape: tuple[int, ...]
    dtype: str  # key into DTYPE_SIZES
    role: str = "activation"
    scale: tuple[float, ...] = ()  # () for float32; len 1 = per-tensor quant; len N = per-channel
    zero_point: tuple[int, ...] = ()
    data: Optional[Any] = None  # numpy array for weight/bias/const tensors, else None

    def __post_init__(self):
        if self.dtype not in DTYPE_SIZES:
            raise UnsupportedOpError(
                f"tensor '{self.name}': unsupported dtype {self.dtype!r} "
                f"(supported: {', '.join(DTYPE_SIZES)})"
            )
        if self.role not in TENSOR_ROLES:
            raise UnsupportedOpError(f"tensor '{self.name}': unknown role {self.role!r}")

    @property
    def size(self) -> int:
        """Element count (product of shape dims)."""
        return _prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.size * DTYPE_SIZES[self.dtype]

    @property
    def is_quantized(self) -> bool:
        return self.dtype in ("int8", "uint8") and len(self.scale) > 0

    @property
    def is_per_channel(self) -> bool:
        return len(self.scale) > 1

    def quant_params(self) -> tuple[float, int]:
        """Per-tensor (scale, zero_point) -- raises if this tensor is per-channel."""
        if self.is_per_channel:
            raise UnsupportedOpError(f"tensor '{self.name}' is per-channel quantized; use scale[i]/zero_point[i]")
        if not self.scale:
            raise UnsupportedOpError(f"tensor '{self.name}' has no quantization parameters")
        return self.scale[0], self.zero_point[0]


@dataclass
class Node:
    op: str
    name: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.op not in SUPPORTED_OPS:
            raise UnsupportedOpError(
                f"node '{self.name}': op {self.op!r} is not in EdgeForge's supported op set "
                f"({', '.join(SUPPORTED_OPS)})"
            )


@dataclass
class ModelIR:
    kind: str  # "classical" | "deep" -- the tier a board's model_tier is checked against
    task: str  # "classification" | "regression"
    input_spec: TensorSpec
    output_spec: TensorSpec
    tensors: dict[str, TensorSpec]
    nodes: list[Node]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ("classical", "deep"):
            raise UnsupportedOpError(f"ModelIR.kind must be 'classical' or 'deep', got {self.kind!r}")
        if self.task not in ("classification", "regression"):
            raise UnsupportedOpError(f"ModelIR.task must be 'classification' or 'regression', got {self.task!r}")
        for name in (self.input_spec.name, self.output_spec.name):
            if name not in self.tensors:
                raise UnsupportedOpError(f"ModelIR: tensor '{name}' referenced but not present in .tensors")

    @property
    def num_classes(self) -> int:
        return self.output_spec.size if self.task == "classification" else 0

    def weight_tensors(self) -> list[TensorSpec]:
        return [t for t in self.tensors.values() if t.role in ("weight", "bias")]

    def activation_tensors(self) -> list[TensorSpec]:
        """Intermediate tensors that need a scratch buffer (not the model input/output)."""
        return [
            t
            for t in self.tensors.values()
            if t.role == "activation" and t.name not in (self.input_spec.name, self.output_spec.name)
        ]

    def param_count(self) -> int:
        return sum(t.size for t in self.weight_tensors())

    def weight_bytes(self) -> int:
        """Total constant (weight/bias) storage -- this is what lands in flash/ROM."""
        return sum(t.nbytes for t in self.weight_tensors())

    def activation_bytes(self) -> int:
        """Total scratch-buffer storage for intermediate activations -- this is what lands in RAM,
        on top of the input/output buffers (also RAM) accounted separately by the caller."""
        return sum(t.nbytes for t in self.activation_tensors())

    def io_bytes(self) -> int:
        return self.input_spec.nbytes + self.output_spec.nbytes

    def describe(self) -> str:
        lines = [
            f"kind={self.kind} task={self.task} params={self.param_count()} "
            f"weight_bytes={self.weight_bytes()} activation_bytes={self.activation_bytes()}",
            f"input:  {self.input_spec.name} shape={self.input_spec.shape} dtype={self.input_spec.dtype}",
            f"output: {self.output_spec.name} shape={self.output_spec.shape} dtype={self.output_spec.dtype}",
        ]
        for n in self.nodes:
            lines.append(f"  {n.op:<18} {n.name}  in={n.inputs} out={n.outputs}")
        return "\n".join(lines)


def quantize_affine(real_value: float, scale: float, zero_point: int, qmin: int = -128, qmax: int = 127) -> int:
    """Standard TFLite-style affine quantization: q = round(real/scale) + zero_point, clamped."""
    q = int(round(real_value / scale)) + zero_point
    return max(qmin, min(qmax, q))


def dequantize_affine(q_value: int, scale: float, zero_point: int) -> float:
    return scale * (q_value - zero_point)


def compute_asymmetric_quant_params(min_val: float, max_val: float, qmin: int = -128, qmax: int = 127) -> tuple[float, int]:
    """Given an observed float range, compute (scale, zero_point) the same way TFLite's
    post-training quantizer does for an asymmetric int8 range."""
    min_val = min(min_val, 0.0)
    max_val = max(max_val, 0.0)
    if max_val == min_val:
        max_val = min_val + 1e-6
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - round(min_val / scale)
    zero_point = max(qmin, min(qmax, int(zero_point)))
    return scale, zero_point
