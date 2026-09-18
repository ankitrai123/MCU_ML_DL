"""Trains Pipeline(StandardScaler, LogisticRegression) on room_comfort.csv and pickles the
whole pipeline. This is the worked example walked through step by step in
TRAINING_A_MODEL.md -- start there if you're new to training a model in Python.

room_comfort.csv has two raw sensor-style columns (temperature_c, humidity_pct) and a label
(comfort: cold/comfortable/hot) -- exactly the shape of CSV EdgeForge expects you to build
for your own sensor + label.

    python -m edgeforge convert --model examples/trained_models/room_comfort_model.pkl \\
        --board arduino_nano33_ble_sense_rev2 --out ./build/room_comfort_arduino
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "room_comfort.csv")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "trained_models" / "room_comfort_model.pkl")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    X = data[["temperature_c", "humidity_pct"]].to_numpy()
    y = data["comfort"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ])
    model.fit(X_train, y_train)
    print(f"accuracy on held-out test data: {model.score(X_test, y_test):.0%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
