"""Trains a small MLPClassifier (relu hidden layers) on iris and pickles it.

    python -m edgeforge convert --model examples/trained_models/iris_mlp.pkl \\
        --board stm32f411 --out ./build/iris_mlp_stm32
"""

import argparse
import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "trained_models" / "iris_mlp.pkl")
    parser.add_argument("--hidden", type=int, nargs="+", default=[8, 5])
    args = parser.parse_args()

    X, y = load_iris(return_X_y=True)
    model = MLPClassifier(hidden_layer_sizes=tuple(args.hidden), activation="relu", max_iter=2000, random_state=0)
    model.fit(X, y)
    print(f"train accuracy: {model.score(X, y):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
