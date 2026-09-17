import pickle
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def boards_dir() -> Path:
    return REPO_ROOT / "boards"


@pytest.fixture(scope="session")
def iris_data():
    from sklearn.datasets import load_iris

    return load_iris(return_X_y=True)


@pytest.fixture(scope="session")
def tree_clf_path(tmp_path_factory, iris_data):
    from sklearn.tree import DecisionTreeClassifier

    X, y = iris_data
    model = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    path = tmp_path_factory.mktemp("models") / "tree_clf.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


@pytest.fixture(scope="session")
def logreg_bin_path(tmp_path_factory, iris_data):
    from sklearn.linear_model import LogisticRegression

    X, y = iris_data
    y_bin = (y == 2).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y_bin)
    path = tmp_path_factory.mktemp("models") / "logreg_bin.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


@pytest.fixture(scope="session")
def mlp_clf_path(tmp_path_factory, iris_data):
    from sklearn.neural_network import MLPClassifier

    X, y = iris_data
    model = MLPClassifier(hidden_layer_sizes=(6,), activation="relu", max_iter=1000, random_state=0).fit(X, y)
    path = tmp_path_factory.mktemp("models") / "mlp_clf.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


@pytest.fixture(scope="session")
def keras_cnn_path(tmp_path_factory):
    tf = pytest.importorskip("tensorflow")
    size = 8
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 0.15, size=(120, size, size, 1)).astype(np.float32)
    y = rng.integers(0, 3, size=(120,))
    half = size // 2
    for i in range(120):
        r0 = 0 if y[i] != 2 else half
        cy, cx = r0 + rng.integers(1, half - 1), rng.integers(1, size - 1)
        yy, xx = np.ogrid[:size, :size]
        X[i, :, :, 0] = np.clip(X[i, :, :, 0] + np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 4.0), 0.0, 1.0)

    tf.random.set_seed(0)
    inputs = tf.keras.Input(shape=(size, size, 1))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.DepthwiseConv2D(3, padding="valid", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, x)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(X, y, epochs=3, verbose=0)

    path = tmp_path_factory.mktemp("models") / "cnn.keras"
    model.save(path)
    return path


@pytest.fixture(scope="session")
def onnx_mlp_path(tmp_path_factory):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper, TensorProto

    rng = np.random.default_rng(0)
    n_in, n_hidden, n_out = 4, 6, 3
    W1 = rng.normal(size=(n_hidden, n_in)).astype(np.float32)
    b1 = rng.normal(size=(n_hidden,)).astype(np.float32)
    W2 = rng.normal(size=(n_out, n_hidden)).astype(np.float32)
    b2 = rng.normal(size=(n_out,)).astype(np.float32)

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, n_in])
    outp = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, n_out])
    inits = [
        numpy_helper.from_array(W1, "W1"), numpy_helper.from_array(b1, "b1"),
        numpy_helper.from_array(W2, "W2"), numpy_helper.from_array(b2, "b2"),
    ]
    nodes = [
        helper.make_node("Gemm", ["input", "W1", "b1"], ["z1"], transB=1),
        helper.make_node("Relu", ["z1"], ["h1"]),
        helper.make_node("Gemm", ["h1", "W2", "b2"], ["z2"], transB=1),
        helper.make_node("Softmax", ["z2"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "mlp", [inp], [outp], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)

    path = tmp_path_factory.mktemp("models") / "mlp.onnx"
    onnx.save(model, str(path))
    return path


@pytest.fixture(scope="session")
def keras_mlp_path(tmp_path_factory, iris_data):
    tf = pytest.importorskip("tensorflow")
    X, y = iris_data
    X = X.astype(np.float32)

    tf.random.set_seed(0)
    inputs = tf.keras.Input(shape=(4,))
    x = tf.keras.layers.Dense(6, activation="relu")(inputs)
    x = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, x)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(X, y, epochs=5, verbose=0)

    path = tmp_path_factory.mktemp("models") / "iris_mlp.keras"
    model.save(path)
    return path
