"""Record a bounded LSL session to a private local CSV, with correct channel labels."""

import argparse
import csv
import math
from pathlib import Path

from .features import CHANNELS
from .stream import open_stream, read_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--source-id")
    parser.add_argument("--channel-order")
    args = parser.parse_args()
    if args.seconds <= 0 or not math.isfinite(args.seconds):
        parser.error("--seconds must be positive and finite")
    inlet = None
    try:
        if args.output.exists():
            raise ValueError("Output already exists; choose a new recording filename")
        inlet, indices, rate = open_stream(args.source_id, args.channel_order)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Recording at {rate:g} Hz. This CSV contains personal biosignal data.", flush=True)
        with args.output.open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Timestamps", *CHANNELS])
            previous_timestamp = None
            for _ in range(max(1, round(args.seconds * rate))):
                values, timestamp = read_sample(inlet, indices)
                if previous_timestamp is not None and not 0 < timestamp - previous_timestamp <= 5 / rate:
                    raise ValueError("Discontinuous timestamps; incomplete recording saved for inspection")
                previous_timestamp = timestamp
                writer.writerow([timestamp, *values])
    except KeyboardInterrupt:
        print("\nStopped; samples already received are saved locally.")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Recording stopped: {exc}\n")
    finally:
        if inlet is not None:
            inlet.close_stream()


if __name__ == "__main__":
    main()
