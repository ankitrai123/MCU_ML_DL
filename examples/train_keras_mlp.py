"""Trains a small Keras dense MLP on iris and saves it as .keras.

Demonstrates EdgeForge's neural-net path: TFLite conversion + post-training
int8 quantization, targeting a "deep" tier board.

    python -m edgeforge convert --model examples/trained_models/iris_mlp.keras \\
        --board stm32f411 --out ./build/iris_mlp_keras_stm32 \\
        --sample-range -3,9
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "trained_models" / "iris_mlp.keras")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    X, y = load_iris(return_X_y=True)
    X = X.astype(np.float32)

    tf.random.set_seed(0)
    inputs = tf.keras.Input(shape=(4,))
    x = tf.keras.layers.Dense(8, activation="relu")(inputs)
    x = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, x)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=args.epochs, verbose=0)
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"train accuracy: {acc:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"wrote {args.out}")
    print("note: iris features range roughly [-2, 8]; pass --sample-range -3,9 to `edgeforge convert`")


if __name__ == "__main__":
    main()
