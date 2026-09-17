"""C source formatting helpers shared by every Jinja2 template.

Buffer naming: every tensor's C storage identifier is ``ef_<sanitized name>``
except the two ends of the ``model_infer`` call, which use the fixed
parameter names ``input``/``output`` directly (the caller's memory, not a
static buffer EdgeForge owns) -- the ``ef_`` prefix exists precisely so an
IR tensor that happens to be named "output" (every classical ingester's
final tensor) can never collide with the C parameter of the same name.
"""

from __future__ import annotations

import numpy as np

from edgeforge.boards.schema import BoardProfile
from edgeforge.ir import TensorSpec

_C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "int", "long", "register", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef", "union", "unsigned", "void",
    "volatile", "while",
}


def sanitize_ident(name: str) -> str:
    out = "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    if not out or out[0].isdigit():
        out = "_" + out
    if out in _C_KEYWORDS:
        out = out + "_"
    return out


def buf_name(tensor_name: str) -> str:
    return "ef_" + sanitize_ident(tensor_name)


def float_literal(x: float) -> str:
    if x != x:  # NaN guard: can't occur from a trained model's real weights, but never emit invalid C silently
        raise ValueError("cannot emit NaN as a C float literal")
    s = f"{float(x):.9g}"
    if "." not in s and "e" not in s and "E" not in s and "inf" not in s and "nan" not in s:
        s += ".0"  # C requires a decimal point or exponent before an 'f' suffix ("2f" is not a float literal)
    return s + "f"


def int_literal(x: int) -> str:
    return str(int(x))


def hex32(x: int) -> str:
    return f"0x{int(x):08X}"


def _scalar_literal(x, dtype: str) -> str:
    return float_literal(x) if dtype == "float32" else int_literal(x)


def c_array_literal(data: np.ndarray, dtype: str, shape=None, per_line: int = 12, indent: str = "    ") -> str:
    """Render `data` as a C initializer list, brace-nested to match `shape` exactly.

    SDCC (unlike gcc) rejects a flat initializer list for a multi-dimensional array --
    "initialization needs curly braces" -- so a 2+-D tensor always gets fully nested
    braces per dimension here rather than relying on C's (gcc-tolerated) brace elision.
    """
    arr = np.asarray(data)
    shape = tuple(shape) if shape is not None else arr.shape
    flat = arr.reshape(-1)
    items = [_scalar_literal(v, dtype) for v in flat.tolist()]
    if not items:
        return "{0}"  # C forbids an empty initializer; a zero-length model tensor can't occur, but stay valid C
    if len(shape) <= 1:
        lines = [indent + ", ".join(items[i : i + per_line]) + "," for i in range(0, len(items), per_line)]
        return "{\n" + "\n".join(lines) + "\n" + indent[:-4] + "}"
    return _nested_literal(items, shape, level=1, indent_unit=indent)


def _nested_literal(items: list, shape: tuple, level: int, indent_unit: str) -> str:
    pad = indent_unit * level
    if len(shape) == 1:
        return "{" + ", ".join(items) + "}"
    dim, rest = shape[0], shape[1:]
    chunk = shape_product(rest)
    parts = [pad + _nested_literal(items[i * chunk : (i + 1) * chunk], rest, level + 1, indent_unit) for i in range(dim)]
    return "{\n" + ",\n".join(parts) + "\n" + indent_unit * (level - 1) + "}"


def shape_product(shape) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def c_dims(shape) -> str:
    """`(4, 4, 3)` -> `[4][4][3]`, `()` -> `[1]` (a scalar still needs one C array dimension)."""
    if not shape:
        return "[1]"
    return "".join(f"[{d}]" for d in shape)


def flat_ref(tensor: TensorSpec, ir, elem_ctype: str) -> str:
    """Like tensor_ref, but always safe to subscript with a single flat index.

    A "linear" node's input can be a multi-dimensional (H,W,C) activation tensor when
    it follows an elided Flatten (row-major C memory is already flat, so Flatten never
    gets its own node -- see keras_ingest's module docstring) -- but that tensor's C
    buffer is still *declared* with its real shape, e.g. `int8_t buf[2][2][4]`, where
    `buf[j]` is a sub-array, not the j'th scalar element. Casting to a flat pointer
    first makes single-index access well-defined regardless of the tensor's declared
    dimensionality.
    """
    ref = tensor_ref(tensor, ir)
    if len(tensor.shape) <= 1:
        return ref
    return f"(({elem_ctype}*){ref})"


def tensor_ref(tensor: TensorSpec, ir) -> str:
    """The C expression a node macro uses to read/write a tensor's storage: the
    model's own input tensor is the function parameter `input` (the caller's memory);
    every other tensor -- including the model's *output* tensor -- gets its own
    `ef_`-prefixed static buffer, so a final copy/dequantize loop can fill the actual
    `output` function parameter without a naming collision against a same-named IR tensor."""
    if tensor.name == ir.input_spec.name:
        return "input"
    return buf_name(tensor.name)


def board_ctype_for(tensor: TensorSpec, board: BoardProfile) -> str:
    """Map a tensor's IR dtype+role to the board-configured C type for that role.
    Every board's YAML supplies all of these roles (schema-enforced), so the same
    mapping logic applies unchanged to a 32-bit ARM board and an 8-bit 8051 board."""
    if tensor.dtype == "float32":
        return board.type_of("real")
    if tensor.dtype == "int8":
        return board.type_of("weight_i8") if tensor.role == "weight" else board.type_of("activation_i8")
    if tensor.dtype in ("int32", "uint32"):
        return board.type_of("bias_i32")
    if tensor.dtype == "uint8":
        return board.type_of("activation_i8")
    raise ValueError(f"tensor '{tensor.name}': no board C-type mapping for dtype {tensor.dtype!r}")
