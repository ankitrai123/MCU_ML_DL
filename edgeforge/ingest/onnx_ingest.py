"""ONNX -> ModelIR, scoped to plain feedforward graphs (Gemm / MatMul+Add chains).

This deliberately targets the same territory as sklearn's MLP path -- "a
small MLP" exported from PyTorch, skl2onnx, etc. -- not arbitrary ONNX graphs.
It reuses the classical (float32) "linear" + "activation" IR ops verbatim, so
it needs zero new codegen: an ONNX MLP and a sklearn MLPClassifier render
through the exact same templates. Convolutional/attention/RNN ONNX graphs are
out of scope for Phase 1 (no TFLite-equivalent int8 quantization path exists
for ONNX here); they raise UnsupportedOpError by op name rather than silently
producing wrong code. A future phase could route ONNX through TFLite's own
onnx->tf->tflite conversion for the deep tier, but that pulls in a fragile
multi-hop conversion chain this phase doesn't need for "small MLP" coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from edgeforge.errors import UnsupportedModelError, UnsupportedOpError
from edgeforge.ingest.base import IngestResult, uniform_range_sampler
from edgeforge.ir import ModelIR, Node, TensorSpec

_ELEMENTWISE_ACTIVATIONS = {"Relu": "relu", "Identity": "identity"}
_FINAL_ONLY_ACTIVATIONS = {"Softmax": "softmax", "Sigmoid": "logistic", "Tanh": "tanh"}
_DEFAULT_SAMPLE_RANGE = (-3.0, 3.0)


def can_handle(path: Path) -> bool:
    return path.suffix.lower() == ".onnx"


def _onnx():
    try:
        import onnx
    except ImportError as e:
        raise UnsupportedModelError(
            "ingesting a .onnx model requires the 'onnx' package, which is not installed. "
            "Install it with: pip install onnx onnxruntime"
        ) from e
    return onnx


def _ort():
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise UnsupportedModelError(
            "ingesting a .onnx model requires 'onnxruntime' (for golden-vector reference inference), "
            "which is not installed. Install it with: pip install onnx onnxruntime"
        ) from e
    return ort


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            onnx = _onnx()
            return onnx.helper.get_attribute_value(a)
    return default


def ingest(path: Path, sample_range: Optional[tuple[float, float]] = None, task: str = "auto") -> IngestResult:
    onnx = _onnx()
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph = model.graph

    initializer_names = {init.name for init in graph.initializer}
    graph_inputs = [i for i in graph.input if i.name not in initializer_names]
    if len(graph_inputs) != 1 or len(graph.output) != 1:
        raise UnsupportedModelError(
            f"'{path}' has {len(graph_inputs)} input(s) and {len(graph.output)} output(s); "
            "EdgeForge's ONNX path supports a single-input, single-output feedforward graph."
        )

    initializers = {init.name: onnx.numpy_helper.to_array(init).astype(np.float64) for init in graph.initializer}
    input_name_onnx = graph_inputs[0].name
    input_dim = graph_inputs[0].type.tensor_type.shape.dim
    n_features = None
    for d in input_dim:
        if d.dim_value:
            n_features = d.dim_value
    if n_features is None:
        raise UnsupportedModelError(
            f"'{path}' input '{input_name_onnx}' has no static feature dimension; "
            "EdgeForge needs a fully static input shape to size C buffers ahead of time."
        )

    input_spec = TensorSpec("input", (n_features,), "float32", role="input")
    tensors: dict[str, TensorSpec] = {"input": input_spec}
    nodes: list[Node] = []
    value_alias: dict[str, str] = {input_name_onnx: "input"}

    def resolve(onnx_name: str) -> str:
        return value_alias.get(onnx_name, onnx_name)

    onnx_nodes = list(graph.node)
    i = 0
    last_op_kind = None  # 'linear' | 'relu' | 'softmax' | 'logistic' | 'tanh' | 'identity'
    layer_idx = 0

    while i < len(onnx_nodes):
        node = onnx_nodes[i]
        op = node.op_type

        if op == "Gemm":
            in_name = resolve(node.input[0])
            w_raw = initializers.get(node.input[1])
            if w_raw is None:
                raise UnsupportedOpError(f"Gemm node '{node.name}': weight input must be a constant initializer")
            b_raw = initializers.get(node.input[2]) if len(node.input) > 2 else None
            alpha = _attr(node, "alpha", 1.0)
            beta = _attr(node, "beta", 1.0)
            trans_a = _attr(node, "transA", 0)
            trans_b = _attr(node, "transB", 0)
            if trans_a:
                raise UnsupportedOpError(f"Gemm node '{node.name}': transA=1 is not supported")
            w = w_raw if trans_b else w_raw.T  # want (n_out, n_in); transB=1 means B is already (out,in)
            w = alpha * w
            b = (beta * b_raw) if b_raw is not None else np.zeros(w.shape[0])
            out_name = _emit_linear(tensors, nodes, layer_idx, in_name, w, b)
            value_alias[node.output[0]] = out_name
            last_op_kind = "linear"
            layer_idx += 1
            i += 1
            continue

        if op == "MatMul":
            in_name = resolve(node.input[0])
            w_raw = initializers.get(node.input[1])
            if w_raw is None:
                raise UnsupportedOpError(f"MatMul node '{node.name}': weight input must be a constant initializer")
            w = w_raw.T  # ONNX MatMul convention: y = x @ B, B is (in,out) -> IR wants (out,in)
            b = np.zeros(w.shape[0])
            consumed = 1
            if i + 1 < len(onnx_nodes) and onnx_nodes[i + 1].op_type == "Add":
                add_node = onnx_nodes[i + 1]
                bias_operand = next((initializers[n] for n in add_node.input if n in initializers), None)
                matmul_out_as_add_input = node.output[0] in add_node.input
                if bias_operand is not None and matmul_out_as_add_input and bias_operand.shape == (w.shape[0],):
                    b = bias_operand
                    node = add_node  # so node.output[0] below refers to the Add's output
                    consumed = 2
            out_name = _emit_linear(tensors, nodes, layer_idx, in_name, w, b)
            value_alias[node.output[0]] = out_name
            last_op_kind = "linear"
            layer_idx += 1
            i += consumed
            continue

        if op in _ELEMENTWISE_ACTIVATIONS:
            in_name = resolve(node.input[0])
            kind = _ELEMENTWISE_ACTIVATIONS[op]
            if kind != "identity":
                nodes.append(Node("activation", f"act{len(nodes)}", [in_name], [in_name], {"kind": kind}))
            value_alias[node.output[0]] = in_name
            last_op_kind = kind
            i += 1
            continue

        if op in _FINAL_ONLY_ACTIVATIONS:
            in_name = resolve(node.input[0])
            kind = _FINAL_ONLY_ACTIVATIONS[op]
            value_alias[node.output[0]] = in_name  # tentatively elided; may be materialized below if not final
            last_op_kind = kind
            i += 1
            continue

        raise UnsupportedOpError(
            f"ONNX op '{op}' is not in EdgeForge's supported feedforward op set "
            "(Gemm, MatMul(+Add), Relu, Identity, and a final Softmax/Sigmoid/Tanh)."
        )

    graph_output_name = graph.output[0].name
    final_name = resolve(graph_output_name)

    if task == "auto":
        task = "classification" if last_op_kind in ("softmax", "logistic") else "regression"

    if task == "classification":
        if tensors[final_name].size == 1:
            raise UnsupportedOpError(
                "model ends in a single-unit output feeding classification; EdgeForge needs an "
                "argmax-able output. Export a >=2-unit final layer (softmax-style) instead of a "
                "single sigmoid unit."
            )
        # Softmax/logistic/tanh right before classification output is elided (argmax-invariant);
        # anything mid-graph already got rejected above since only the *final* op alias applies here.
    else:
        if last_op_kind in ("logistic", "tanh"):
            raise UnsupportedOpError(
                f"regression output activation '{last_op_kind}' needs a transcendental function EdgeForge's "
                "freestanding codegen doesn't link; export with a linear (no-op) final activation."
            )

    if final_name != "output":
        tensors["output"] = TensorSpec("output", tensors[final_name].shape, "float32", role="output")
        for n in nodes:
            n.outputs = ["output" if o == final_name else o for o in n.outputs]
            n.inputs = ["output" if o == final_name else o for o in n.inputs]
        del tensors[final_name]

    output_spec = tensors["output"]
    ir = ModelIR(
        kind="classical",
        task=task,
        input_spec=input_spec,
        output_spec=output_spec,
        tensors=tensors,
        nodes=nodes,
        metadata={"source_framework": "onnx", "model_class": "onnx_feedforward", "n_layers": layer_idx},
    )

    ort = _ort()
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    ort_input_name = session.get_inputs()[0].name
    ort_output_name = session.get_outputs()[0].name

    def reference_fn(sample: np.ndarray):
        x = sample.reshape(1, -1).astype(np.float32)
        out = session.run([ort_output_name], {ort_input_name: x})[0].reshape(-1).astype(np.float32)
        if task == "classification":
            return int(np.argmax(out)), out
        return None, out

    lo, hi = sample_range or _DEFAULT_SAMPLE_RANGE
    input_sampler = uniform_range_sampler(np.full(n_features, lo), np.full(n_features, hi))

    return IngestResult(ir=ir, reference_fn=reference_fn, input_sampler=input_sampler, source_kind="onnx")


def _emit_linear(tensors, nodes, layer_idx, in_name, w, b) -> str:
    n_out, n_in = w.shape
    w_name, b_name, out_name = f"w{layer_idx}", f"b{layer_idx}", f"onnx_linear{layer_idx}"
    tensors[w_name] = TensorSpec(w_name, (n_out, n_in), "float32", role="weight", data=w)
    tensors[b_name] = TensorSpec(b_name, (n_out,), "float32", role="bias", data=b)
    tensors[out_name] = TensorSpec(out_name, (n_out,), "float32", role="activation")
    nodes.append(
        Node("linear", f"fc{layer_idx}", [in_name], [out_name], {"weight": w_name, "bias": b_name, "n_in": n_in, "n_out": n_out})
    )
    return out_name
