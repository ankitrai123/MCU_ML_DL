"""Trains a small DecisionTreeClassifier on iris and pickles it.

This is the model used by EdgeForge's 8051 definition-of-done:

    python -m edgeforge convert --model examples/trained_models/iris_tree.pkl \\
        --board 8051_at89s52 --out ./build/iris_tree_8051
"""

import argparse
import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "trained_models" / "iris_tree.pkl")
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()

    X, y = load_iris(return_X_y=True)
    model = DecisionTreeClassifier(max_depth=args.max_depth, random_state=0)
    model.fit(X, y)
    print(f"train accuracy: {model.score(X, y):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
