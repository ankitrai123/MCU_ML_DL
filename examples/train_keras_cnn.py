"""Trains a small Keras CNN (Conv2D -> MaxPool2D -> DepthwiseConv2D -> Dense) on a
synthetic "which corner is the bright blob in" image task, and saves it as .keras.

Exercises EdgeForge's conv2d/depthwise_conv2d/maxpool2d codegen path, not just
plain dense layers.

    python -m edgeforge convert --model examples/trained_models/blob_cnn.keras \\
        --board stm32f411 --out ./build/blob_cnn_stm32 --sample-range 0,1
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def _make_dataset(n: int, size: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 0.15, size=(n, size, size, 1)).astype(np.float32)
    y = rng.integers(0, 4, size=(n,))
    half = size // 2
    for i in range(n):
        cls = y[i]
        r0 = 0 if cls in (0, 1) else half
        c0 = 0 if cls in (0, 2) else half
        cy = r0 + rng.integers(1, half - 2)
        cx = c0 + rng.integers(1, half - 2)
        yy, xx = np.ogrid[:size, :size]
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 6.0)
        X[i, :, :, 0] = np.clip(X[i, :, :, 0] + blob, 0.0, 1.0)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "trained_models" / "blob_cnn.keras")
    parser.add_argument("--size", type=int, default=12, help="image side length")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    Xtr, ytr = _make_dataset(800, args.size, seed=0)
    Xte, yte = _make_dataset(200, args.size, seed=1)

    tf.random.set_seed(0)
    inputs = tf.keras.Input(shape=(args.size, args.size, 1))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.DepthwiseConv2D(3, padding="valid", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs, x)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, epochs=args.epochs, verbose=0)
    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"test accuracy: {acc:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
