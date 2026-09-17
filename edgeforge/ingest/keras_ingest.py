"""Keras (.h5/.keras) -> TFLite post-training int8 quantization -> ModelIR.

Extraction reads the TFLite flatbuffer's own object-API model (
``tensorflow.lite.python.schema_py_generated.ModelT``) rather than the
``Interpreter``'s private introspection hooks or (worse) the CPU delegate's
rewritten graph -- this is the same static structure ``tflite_convert`` and
TFLite Micro's own `interpreter` read, so it stays correct across ops we
don't special-case.

Supported ops: FULLY_CONNECTED, CONV_2D, DEPTHWISE_CONV_2D, MAX_POOL_2D,
AVERAGE_POOL_2D, SOFTMAX (elided for classification -- see sklearn_ingest's
docstring for why argmax makes that safe), with RELU/RELU6 fused activations.
RESHAPE/SQUEEZE/EXPAND_DIMS and the SHAPE/STRIDED_SLICE/PACK cluster Keras
emits for a dynamic-shape Flatten() are transparent: they only ever compute
*which* shape to reshape to, never touch real activation data, and a flat
row-major C buffer needs no reshaping at all -- so ingest walks through them
via a tensor-alias map instead of emitting IR nodes for them. Anything else
(LSTM, attention, BatchNorm as a standalone op, etc.) raises UnsupportedOpError
by name rather than silently emitting wrong code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from edgeforge.errors import UnsupportedModelError, UnsupportedOpError
from edgeforge.ingest.base import IngestResult, uniform_range_sampler
from edgeforge.ir import ModelIR, Node, TensorSpec

_ALIAS_THROUGH_OPS = {"RESHAPE", "SQUEEZE", "EXPAND_DIMS"}
_SHAPE_METADATA_OPS = {"SHAPE", "STRIDED_SLICE", "PACK", "CONCATENATION"}
_POOL_OPS = {"MAX_POOL_2D": "max", "AVERAGE_POOL_2D": "avg"}

_TFLITE_DTYPE_TO_IR = {9: "int8", 2: "int32", 0: "float32", 3: "uint8"}


def can_handle(path: Path) -> bool:
    return path.suffix.lower() in (".h5", ".keras")


def _tf():
    try:
        import tensorflow as tf
    except ImportError as e:
        raise UnsupportedModelError(
            "ingesting a .h5/.keras model requires TensorFlow, which is not installed. "
            "Install it with: pip install tensorflow-cpu"
        ) from e
    return tf


def _schema_module():
    from tensorflow.lite.python import schema_py_generated as sch

    return sch


def ingest(
    path: Path,
    representative_data: Optional[np.ndarray] = None,
    sample_range: Optional[tuple[float, float]] = None,
    task: str = "auto",
) -> IngestResult:
    tf = _tf()
    model = tf.keras.models.load_model(path)

    if isinstance(model.input_shape, list) or isinstance(model.output_shape, list):
        raise UnsupportedModelError(
            f"'{path}' has multiple inputs or outputs; EdgeForge's neural-net path supports a "
            "single-input, single-output Keras model (Sequential or single-branch Functional)."
        )

    input_shape = tuple(d for d in model.input_shape[1:] if d is not None)
    if any(d is None for d in model.input_shape[1:]):
        raise UnsupportedModelError(
            f"'{path}' has a dynamic (None) non-batch input dimension {model.input_shape}; "
            "EdgeForge needs a fully static input shape to size C buffers ahead of time."
        )

    lo, hi = sample_range or (0.0, 1.0)
    rng = np.random.default_rng(0)
    if representative_data is None:
        representative_data = rng.uniform(lo, hi, size=(100, *input_shape)).astype(np.float32)
        used_synthetic_rep_data = True
    else:
        used_synthetic_rep_data = False

    tflite_bytes = _convert_to_tflite_int8(tf, model, representative_data, input_shape)
    ir, output_tflite_idx = _parse_tflite_to_ir(tflite_bytes, task, input_shape)
    ir.metadata["source_framework"] = "keras"
    ir.metadata["model_class"] = type(model).__name__
    ir.metadata["used_synthetic_representative_data"] = used_synthetic_rep_data

    reference_fn = _make_reference_fn(tflite_bytes, ir, input_shape, output_tflite_idx)

    flat = representative_data.reshape(representative_data.shape[0], -1)
    span = flat.max(axis=0) - flat.min(axis=0)
    pad = np.where(span > 0, span * 0.15, 1.0)
    input_sampler = uniform_range_sampler(flat.min(axis=0) - pad, flat.max(axis=0) + pad)

    return IngestResult(ir=ir, reference_fn=reference_fn, input_sampler=input_sampler, source_kind="keras")


def _convert_to_tflite_int8(tf, model, representative_data: np.ndarray, input_shape: tuple[int, ...]) -> bytes:
    def rep_gen():
        for i in range(representative_data.shape[0]):
            yield [representative_data[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        return converter.convert()
    except Exception as e:  # TFLiteConverter raises a mix of RuntimeError/ValueError/Convert errors
        raise UnsupportedModelError(
            f"TFLite int8 conversion failed for this Keras model: {e}\n"
            "This usually means the model uses a layer TFLite's int8 quantizer can't fully cover "
            "(check TFLITE_BUILTINS_INT8 op coverage for your layer types)."
        ) from e


def _enum_maps(sch):
    code_names = {v: k for k, v in vars(sch.BuiltinOperator).items() if isinstance(v, int)}
    act_names = {v: k for k, v in vars(sch.ActivationFunctionType).items() if isinstance(v, int)}
    return code_names, act_names


def _tensor_shape(t, role: str) -> tuple[int, ...]:
    shape = list(t.shape) if t.shape is not None else []
    # Only activation-ish tensors (input/output/intermediate) carry a leading batch dim to drop.
    # A weight's leading dim is structural even when it's 1 -- e.g. DEPTHWISE_CONV_2D's weight
    # layout is (1, filter_h, filter_w, out_channels), where that 1 is not a batch dimension.
    if role not in ("weight", "bias") and shape and shape[0] == 1:
        shape = shape[1:]
    return tuple(int(d) for d in shape)


def _tensor_dtype(t) -> str:
    dt = _TFLITE_DTYPE_TO_IR.get(int(t.type))
    if dt is None:
        raise UnsupportedOpError(f"tensor '{t.name.decode()}': unsupported TFLite tensor type code {t.type}")
    return dt


def _same_padding_before(in_size: int, filt: int, stride: int, out_size: int) -> int:
    """TFLite/TF's SAME-padding formula, given the output size it already computed
    (trusted over re-deriving it, so this can never disagree with the tensor's real shape)."""
    pad_total = max((out_size - 1) * stride + filt - in_size, 0)
    return pad_total // 2


def _parse_tflite_to_ir(tflite_bytes: bytes, task: str, input_shape: tuple[int, ...]) -> ModelIR:
    sch = _schema_module()
    m = sch.ModelT.InitFromPackedBuf(bytearray(tflite_bytes), 0)
    code_names, act_names = _enum_maps(sch)
    sg = m.subgraphs[0]
    fb_tensors = sg.tensors

    tensors: dict[str, TensorSpec] = {}
    alias: dict[int, int] = {}

    def resolve(idx: int) -> int:
        seen = set()
        while idx in alias and idx not in seen:
            seen.add(idx)
            idx = alias[idx]
        return idx

    def tensor_name(idx: int) -> str:
        raw = fb_tensors[idx].name.decode()
        return "t" + str(idx) + "_" + "".join(c if c.isalnum() else "_" for c in raw)[-40:]

    def get_or_build_tensor(idx: int, role: str) -> str:
        idx = resolve(idx)
        name = tensor_name(idx)
        if name in tensors:
            return name
        t = fb_tensors[idx]
        shape = _tensor_shape(t, role)
        dtype = _tensor_dtype(t)
        q = t.quantization
        scale = tuple(float(s) for s in q.scale) if q and q.scale is not None else ()
        zp = tuple(int(z) for z in q.zeroPoint) if q and q.zeroPoint is not None else ()
        buf = m.buffers[t.buffer]
        data = None
        if buf.data is not None and len(buf.data) > 0:
            np_dtype = {"int8": np.int8, "int32": np.int32, "uint8": np.uint8, "float32": np.float32}[dtype]
            data = np.frombuffer(bytes(buf.data), dtype=np_dtype).reshape(shape if shape else (1,))
        tensors[name] = TensorSpec(name, shape, dtype, role=role, scale=scale, zero_point=zp, data=data)
        return name

    input_idx = int(sg.inputs[0])
    output_idx = int(sg.outputs[0])
    input_name = get_or_build_tensor(input_idx, "input")

    nodes: list[Node] = []
    last_real_op_name = None

    for op in sg.operators:
        opcode = m.operatorCodes[op.opcodeIndex]
        op_name = code_names.get(opcode.builtinCode, f"code{opcode.builtinCode}")
        ins = [int(i) for i in op.inputs if int(i) >= 0]
        outs = [int(i) for i in op.outputs]

        if op_name in _SHAPE_METADATA_OPS:
            continue
        if op_name in _ALIAS_THROUGH_OPS:
            alias[outs[0]] = resolve(ins[0])
            continue

        if op_name == "FULLY_CONNECTED":
            _build_fully_connected(op, act_names, ins, outs, get_or_build_tensor, nodes)
        elif op_name == "CONV_2D":
            _build_conv(op, act_names, ins, outs, get_or_build_tensor, tensors, nodes, "conv2d")
        elif op_name == "DEPTHWISE_CONV_2D":
            _build_conv(op, act_names, ins, outs, get_or_build_tensor, tensors, nodes, "depthwise_conv2d")
        elif op_name in _POOL_OPS:
            _build_pool(op, ins, outs, get_or_build_tensor, tensors, nodes, _POOL_OPS[op_name])
        elif op_name == "SOFTMAX":
            in_name = get_or_build_tensor(ins[0], "activation")
            out_name = get_or_build_tensor(outs[0], "activation")
            nodes.append(Node("activation", f"node{len(nodes)}_softmax", [in_name], [out_name], {"kind": "softmax"}))
        else:
            raise UnsupportedOpError(
                f"TFLite op '{op_name}' is not in EdgeForge's supported op set "
                "(FULLY_CONNECTED, CONV_2D, DEPTHWISE_CONV_2D, MAX_POOL_2D, AVERAGE_POOL_2D, SOFTMAX, "
                "plus transparent RESHAPE/SQUEEZE/EXPAND_DIMS)."
            )
        last_real_op_name = op_name

    resolved_output_idx = resolve(output_idx)
    final_tensor_name = tensor_name(resolved_output_idx)

    if task == "auto":
        if last_real_op_name == "SOFTMAX":
            task = "classification"
        elif last_real_op_name in ("FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D"):
            task = "regression"
        else:
            raise UnsupportedModelError(
                f"can't auto-detect task from a model ending in '{last_real_op_name}'; pass task explicitly."
            )

    if task == "classification":
        if last_real_op_name == "SOFTMAX":
            # Elide softmax: argmax is invariant to it, and it needs no int8 LUT this way.
            softmax_node = nodes[-1]
            output_name = softmax_node.inputs[0]
            nodes.pop()
        else:
            output_name = final_tensor_name
        if tensors[output_name].size == 1:
            raise UnsupportedOpError(
                "model ends in a single-unit output feeding classification (sigmoid-binary style); "
                "EdgeForge needs an argmax-able output. Use a 2-unit Dense(..., activation='softmax') "
                "final layer instead of Dense(1, activation='sigmoid')."
            )
        tensors[output_name] = TensorSpec(
            output_name, tensors[output_name].shape, tensors[output_name].dtype, role="output",
            scale=tensors[output_name].scale, zero_point=tensors[output_name].zero_point,
        )
    else:
        output_name = final_tensor_name
        tensors[output_name] = TensorSpec(
            output_name, tensors[output_name].shape, tensors[output_name].dtype, role="output",
            scale=tensors[output_name].scale, zero_point=tensors[output_name].zero_point,
        )

    input_spec = tensors[input_name]
    output_spec = tensors[output_name]
    ir = ModelIR(kind="deep", task=task, input_spec=input_spec, output_spec=output_spec, tensors=tensors, nodes=nodes, metadata={})
    # The tflite tensor index backing ir.output_spec -- tensor_name() embeds it as a "t<idx>_" prefix,
    # so this always resolves to the exact tensor the C code's output corresponds to (the pre-softmax
    # logits when softmax was elided, not whatever the interpreter calls its own official output).
    output_tflite_idx = int(output_name.split("_", 1)[0][1:])
    return ir, output_tflite_idx


def _build_fully_connected(op, act_names, ins, outs, get_or_build_tensor, nodes):
    in_name = get_or_build_tensor(ins[0], "activation")
    w_name = get_or_build_tensor(ins[1], "weight")
    b_name = get_or_build_tensor(ins[2], "bias") if len(ins) > 2 else None
    out_name = get_or_build_tensor(outs[0], "activation")
    fused = act_names.get(getattr(op.builtinOptions, "fusedActivationFunction", 0), "NONE")
    nodes.append(
        Node(
            "linear",
            f"node{len(nodes)}_fc",
            [in_name],
            [out_name],
            {"weight": w_name, "bias": b_name, "fused_activation": fused.lower()},
        )
    )


def _build_conv(op, act_names, ins, outs, get_or_build_tensor, tensors, nodes, op_kind):
    in_name = get_or_build_tensor(ins[0], "activation")
    w_name = get_or_build_tensor(ins[1], "weight")
    b_name = get_or_build_tensor(ins[2], "bias") if len(ins) > 2 else None
    out_name = get_or_build_tensor(outs[0], "activation")
    opts = op.builtinOptions
    fused = act_names.get(getattr(opts, "fusedActivationFunction", 0), "NONE").lower()

    in_h, in_w, in_c = tensors[in_name].shape
    out_h, out_w, out_c = tensors[out_name].shape
    # conv2d weight: (out_ch, filter_h, filter_w, in_ch); depthwise: (1, filter_h, filter_w, out_ch)
    filter_h, filter_w = tensors[w_name].shape[1], tensors[w_name].shape[2]
    is_same = opts.padding == 0
    pad_top = _same_padding_before(in_h, filter_h, opts.strideH, out_h) if is_same else 0
    pad_left = _same_padding_before(in_w, filter_w, opts.strideW, out_w) if is_same else 0

    nodes.append(
        Node(
            op_kind,
            f"node{len(nodes)}_{op_kind}",
            [in_name],
            [out_name],
            {
                "weight": w_name,
                "bias": b_name,
                "in_h": in_h, "in_w": in_w, "in_c": in_c,
                "out_h": out_h, "out_w": out_w, "out_c": out_c,
                "filter_h": filter_h, "filter_w": filter_w,
                "stride_h": int(opts.strideH), "stride_w": int(opts.strideW),
                "pad_top": pad_top, "pad_left": pad_left,
                "depth_multiplier": int(getattr(opts, "depthMultiplier", 1)),
                "fused_activation": fused,
            },
        )
    )


def _build_pool(op, ins, outs, get_or_build_tensor, tensors, nodes, reduce_kind):
    in_name = get_or_build_tensor(ins[0], "activation")
    out_name = get_or_build_tensor(outs[0], "activation")
    opts = op.builtinOptions

    in_h, in_w, in_c = tensors[in_name].shape
    out_h, out_w, out_c = tensors[out_name].shape
    is_same = opts.padding == 0
    pad_top = _same_padding_before(in_h, opts.filterHeight, opts.strideH, out_h) if is_same else 0
    pad_left = _same_padding_before(in_w, opts.filterWidth, opts.strideW, out_w) if is_same else 0

    nodes.append(
        Node(
            "maxpool2d",
            f"node{len(nodes)}_pool",
            [in_name],
            [out_name],
            {
                "reduce": reduce_kind,
                "in_h": in_h, "in_w": in_w, "in_c": in_c,
                "out_h": out_h, "out_w": out_w, "out_c": out_c,
                "filter_h": int(opts.filterHeight), "filter_w": int(opts.filterWidth),
                "stride_h": int(opts.strideH), "stride_w": int(opts.strideW),
                "pad_top": pad_top, "pad_left": pad_left,
            },
        )
    )


def _make_reference_fn(tflite_bytes: bytes, ir: ModelIR, input_shape: tuple[int, ...], output_tflite_idx: int):
    """Runs the real TFLite Interpreter, reading back the exact tensor ir.output_spec
    represents -- the pre-softmax logits when softmax was elided, not the interpreter's
    own official output. Comparing against the *official* (post-softmax) tensor instead
    would occasionally disagree with the generated C at a genuine tie: softmax's own int8
    quantization can round two close-but-unequal logits to the identical output value,
    while the pre-softmax logits (what the C code actually argmaxes) are not tied.
    `experimental_preserve_all_tensors` is what makes a non-output tensor readable after invoke()."""
    tf = _tf()
    interp = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_preserve_all_tensors=True,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    in_scale, in_zp = in_detail["quantization"]
    out_scale, out_zp = ir.output_spec.scale[0], ir.output_spec.zero_point[0]

    def reference_fn(sample: np.ndarray):
        x = sample.reshape((1,) + input_shape).astype(np.float32)
        xq = np.clip(np.round(x / in_scale) + in_zp, -128, 127).astype(np.int8)
        interp.set_tensor(in_detail["index"], xq)
        interp.invoke()
        raw = interp.get_tensor(output_tflite_idx).reshape(-1)
        dequant = (out_scale * (raw.astype(np.int32) - out_zp)).astype(np.float32)
        if ir.task == "classification":
            return int(np.argmax(dequant)), dequant
        return None, dequant

    return reference_fn
