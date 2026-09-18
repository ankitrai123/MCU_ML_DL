"""Trains Pipeline(StandardScaler, LogisticRegression) on iris and pickles the whole
pipeline -- demonstrates EdgeForge folding a fitted scaler into the generated C, so a
raw (un-standardized) sensor-style reading -- e.g. a real petal length in cm, not a
pre-normalized -3..3 value -- gets normalized on-device exactly the way training data
was, instead of requiring you to replicate that math by hand in read_sensor().

    python -m edgeforge convert --model examples/trained_models/iris_logreg_scaled.pkl \\
        --board stm32f411 --out ./build/iris_logreg_scaled_stm32

(No --sample-range needed: EdgeForge derives a sensible raw-value golden-vector sampling
range from the fitted StandardScaler's own mean_/scale_ automatically.)
"""

import argparse
import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "trained_models" / "iris_logreg_scaled.pkl")
    args = parser.parse_args()

    X, y = load_iris(return_X_y=True)  # raw measurements in cm, e.g. sepal length ~4.3-7.9
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X, y)
    print(f"train accuracy: {model.score(X, y):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
