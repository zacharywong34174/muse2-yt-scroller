"""Deterministic synthetic plumbing demo; no real EEG, hardware, or keyboard IO."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .decision import BlinkDecision
from .features import CHANNELS, WINDOW_SAMPLES, FeatureExtract
from .pipeline import classify_window
from .train import fit_model


def synthetic_window(label, rng):
    # Deliberately separable toy signals, not a physiological blink simulator.
    values = rng.normal(0, 3, size=(WINDOW_SAMPLES, 4))
    pulse = np.exp(-0.5 * ((np.arange(WINDOW_SAMPLES) - 50) / 9) ** 2)
    if label in ("LB", "RB"):
        values[:, 0 if label == "LB" else 1] += rng.uniform(90, 120) * pulse
    return pd.DataFrame(values, columns=CHANNELS)


def make_training_data(seed=42, per_class=12):
    rng = np.random.default_rng(seed)
    return [(label, synthetic_window(label, rng))
            for label in ("LB", "NB", "RB") for _ in range(per_class)]


def run_demo():
    training = make_training_data()
    features = pd.concat([FeatureExtract(df=w).frame() for _, w in training], ignore_index=True)
    bundle = fit_model(features, [label for label, _ in training])
    classifier = bundle["classifier"]
    decision = BlinkDecision(classifier.classes_)
    rng = np.random.default_rng(2026)
    actions = []
    print("SYNTHETIC DEMO — no headset, personal recordings, or keyboard events")
    for expected in ["NB"] * 2 + ["LB"] * 5 + ["NB"] * 2 + ["RB"] * 5 + ["NB"] * 2:
        label, _, action = classify_window(classifier, decision, synthetic_window(expected, rng))
        print(f"toy input={expected}  prediction={label}  action={action or '-'}")
        if action:
            actions.append(action)
    if actions != ["pageup", "pagedown"]:
        raise RuntimeError(f"Synthetic smoke check failed: {actions}")
    print("PASS: pageup, pagedown. This demonstrates software flow, not real-world accuracy.")
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-training", type=Path,
                        help="Optionally write synthetic CSVs to a new local directory")
    args = parser.parse_args()
    if args.write_training is not None:
        if args.write_training.exists():
            parser.exit(2, "Choose a new output directory; refusing to overwrite existing data.\n")
        for index, (label, window) in enumerate(make_training_data()):
            folder = args.write_training / label
            folder.mkdir(parents=True, exist_ok=True)
            window.to_csv(folder / f"synthetic_{index:03d}.csv", index=False)
        print(f"Synthetic training windows written to {args.write_training}")
    run_demo()


if __name__ == "__main__":
    main()
