"""Extract labelled windows using explicit zero-based start sample indices."""

import argparse
from pathlib import Path

import pandas as pd

from .features import CHANNELS, WINDOW_SAMPLES, FeatureExtract


def prepare(recording, annotations, output):
    if output.exists():
        raise ValueError("Output directory exists; choose a new path to avoid mixing sessions")
    data = pd.read_csv(recording)
    labels = pd.read_csv(annotations)
    if not {"label", "start_sample"}.issubset(labels.columns) or labels.empty:
        raise ValueError("Annotations need label,start_sample columns and at least one row")
    windows = []
    for row in labels.itertuples(index=False):
        if row.label not in ("LB", "NB", "RB"):
            raise ValueError("Annotation labels must be LB, NB, or RB")
        start = int(row.start_sample)
        if start < 0 or start != float(row.start_sample) or start + WINDOW_SAMPLES > len(data):
            raise ValueError("Annotation start_sample must locate a complete 100-sample window")
        window = data.iloc[start:start + WINDOW_SAMPLES]
        FeatureExtract(df=window).frame()  # Validate before writing any windows.
        windows.append((row.label, window.loc[:, list(CHANNELS)]))
    for index, (label, window) in enumerate(windows):
        folder = output / label
        folder.mkdir(parents=True, exist_ok=True)
        window.to_csv(folder / f"window_{index:04d}.csv", index=False)
    return len(windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = prepare(args.recording, args.annotations, args.output)
    except (ValueError, OSError, OverflowError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    print(f"Wrote {count} labelled windows to {args.output}")


if __name__ == "__main__":
    main()
