"""Trains a LogisticRegression on iris and pickles it.

    python -m edgeforge convert --model examples/trained_models/iris_logreg.pkl \\
        --board stm32f411 --out ./build/iris_logreg_stm32
"""

import argparse
import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "trained_models" / "iris_logreg.pkl")
    args = parser.parse_args()

    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    print(f"train accuracy: {model.score(X, y):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
