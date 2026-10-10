"""Train a local random forest from labelled, fixed-length CSV windows."""

import argparse
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .features import FEATURE_NAMES, WINDOW_SAMPLES, FeatureExtract


def training_table(data_dir):
    frames, labels = [], []
    for label in ("LB", "NB", "RB"):
        files = sorted((Path(data_dir) / label).glob("*.csv"))
        if not files:
            raise ValueError(f"No CSV windows in {Path(data_dir) / label}")
        for path in files:
            df = pd.read_csv(path)
            if len(df) != WINDOW_SAMPLES:
                raise ValueError(f"{path.name}: expected {WINDOW_SAMPLES} rows, got {len(df)}")
            # Features and labels are created together, eliminating the old
            # dependence on two independent directory enumeration orders.
            frames.append(FeatureExtract(df=df).frame())
            labels.append(label)
    return pd.concat(frames, ignore_index=True), labels


def fit_model(features, labels, sample_rate=256.0):
    if sample_rate <= 0 or not math.isfinite(sample_rate):
        raise ValueError("Sample rate must be positive and finite")
    if set(labels) != {"LB", "NB", "RB"}:
        raise ValueError("Training requires all three labels: LB, NB, RB")
    classifier = RandomForestClassifier(n_estimators=100, random_state=69)
    classifier.fit(features, labels)
    return {"format_version": 1, "classifier": classifier,
            "sample_rate": float(sample_rate), "window_samples": WINDOW_SAMPLES,
            "feature_names": FEATURE_NAMES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/blinks.joblib"))
    parser.add_argument("--sample-rate", type=float, default=256.0,
                        help="Actual rate of the training recordings; must match live acquisition")
    args = parser.parse_args()
    try:
        if args.model.exists():
            raise ValueError("Model already exists; choose a new output path")
        features, labels = training_table(args.data_dir)
        bundle = fit_model(features, labels, args.sample_rate)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.model)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Training failed: {exc}\n")
    print(f"Trained on {len(labels)} labelled windows; model saved to {args.model}")
    print("No accuracy claim: evaluate on a separate recording session before live use.")


if __name__ == "__main__":
    main()
