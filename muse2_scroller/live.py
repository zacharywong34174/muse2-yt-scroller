"""Classify a Muse-compatible LSL EEG stream; keyboard control is opt-in."""

import argparse
import math
from collections import deque
from pathlib import Path
import time

import joblib
import pandas as pd

from .decision import BlinkDecision
from .features import CHANNELS, FEATURE_NAMES, SLIDE_SAMPLES, WINDOW_SAMPLES
from .pipeline import classify_window
from .stream import open_stream, read_sample


def load_model(path):
    if not path.is_file():
        raise ValueError("Model not found; train one locally using muse2_scroller.train")
    # Joblib uses pickle: callers must only load a model they created/trust.
    bundle = joblib.load(path)
    if (not isinstance(bundle, dict) or bundle.get("format_version") != 1
            or bundle.get("window_samples") != WINDOW_SAMPLES
            or tuple(bundle.get("feature_names", ())) != FEATURE_NAMES):
        raise ValueError("Incompatible model; retrain using this release's training command")
    classifier = bundle.get("classifier")
    BlinkDecision(getattr(classifier, "classes_", ()))
    if tuple(getattr(classifier, "feature_names_in_", ())) != FEATURE_NAMES:
        raise ValueError("Model feature names/order do not match this release")
    rate = bundle.get("sample_rate", 0)
    if not isinstance(rate, (int, float)) or rate <= 0 or not math.isfinite(rate):
        raise ValueError("Model sample rate must be positive and finite")
    return bundle


def send_action(action, enable_control=False):
    if action not in ("pageup", "pagedown"):
        raise ValueError("Only pageup and pagedown are supported")
    if enable_control:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.press(action)
    print(f"{'KEY' if enable_control else 'DRY RUN'}: {action}", flush=True)


def run(bundle, source_id=None, channel_order=None, enable_control=False, max_windows=None):
    if enable_control:
        print("Keyboard control starts in 5 seconds. Focus a safe scrollable page.", flush=True)
        print("Stop with Ctrl+C in this terminal or move the pointer to a screen corner.", flush=True)
        time.sleep(5)
    inlet, indices, rate = open_stream(source_id, channel_order)
    try:
        if not math.isclose(rate, bundle["sample_rate"], rel_tol=0.01):
            raise ValueError("Live sample rate differs from training; recollect/retrain at the same rate")
        classifier = bundle["classifier"]
        decision = BlinkDecision(classifier.classes_)
        window = deque(maxlen=WINDOW_SAMPLES)
        previous_timestamp = None
        processed = 0
        since_prediction = 0
        print(f"Streaming {rate:g} Hz; window={WINDOW_SAMPLES}, slide={SLIDE_SAMPLES}; "
              f"{'CONTROL ENABLED' if enable_control else 'dry run (no keys)'}", flush=True)
        while max_windows is None or processed < max_windows:
            sample, timestamp = read_sample(inlet, indices)
            if previous_timestamp is not None:
                gap = timestamp - previous_timestamp
                if gap <= 0 or gap > 5 / rate:
                    raise ValueError("Discontinuous EEG timestamps; restart after checking the stream")
            previous_timestamp = timestamp
            was_full = len(window) == WINDOW_SAMPLES
            window.append(sample)
            if not was_full:
                continue
            # As in the prototype, first predict after the initial window + slide.
            since_prediction += 1
            if since_prediction < SLIDE_SAMPLES:
                continue
            since_prediction = 0
            label, probabilities, action = classify_window(
                classifier, decision, pd.DataFrame(window, columns=CHANNELS))
            print(f"prediction={label} confidence={max(probabilities):.3f}", flush=True)
            if action:
                send_action(action, enable_control)
            processed += 1
    finally:
        inlet.close_stream()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="Trusted local joblib only: loading untrusted pickle files can execute code")
    parser.add_argument("--source-id", help="Select a specific LSL source when several are present")
    parser.add_argument("--channel-order", help="Explicit comma-separated order if metadata is missing")
    parser.add_argument("--enable-control", action="store_true", help="Send OS-wide Page Up/Down keys")
    parser.add_argument("--max-windows", type=int, help="Stop after this many predictions")
    args = parser.parse_args()
    if args.max_windows is not None and args.max_windows <= 0:
        parser.error("--max-windows must be positive")
    try:
        bundle = load_model(args.model)
        run(bundle, args.source_id, args.channel_order, args.enable_control, args.max_windows)
    except KeyboardInterrupt:
        print("\nStopped.")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Stopped: {exc}\n")


if __name__ == "__main__":
    main()
