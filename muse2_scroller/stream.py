"""Small LSL boundary: select one stream and resolve channels by their labels."""

import math

import numpy as np

from .features import CHANNELS


def channel_indices(labels):
    normalized = [str(label).strip().upper() for label in labels]
    if any(normalized.count(channel) != 1 for channel in CHANNELS):
        raise ValueError("LSL channel labels must include AF7, AF8, TP9, TP10 exactly once")
    return [normalized.index(channel) for channel in CHANNELS]


def open_stream(source_id=None, channel_order=None):
    try:
        from pylsl import StreamInlet, resolve_streams
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Install the [live] extra and liblsl; see README hardware setup") from exc
    streams = [s for s in resolve_streams(wait_time=3.0) if s.type() == "EEG"]
    if source_id is not None:
        streams = [s for s in streams if s.source_id() == source_id]
    if len(streams) != 1:
        raise RuntimeError(f"Found {len(streams)} matching EEG streams; start one or use --source-id")
    inlet = StreamInlet(streams[0], max_buflen=1, recover=False)
    try:
        info = inlet.info(timeout=3.0)
        if channel_order is None:
            node = info.desc().child("channels").child("channel")
            labels = []
            for _ in range(info.channel_count()):
                labels.append(node.child_value("label"))
                node = node.next_sibling()
        else:
            labels = channel_order.split(",")
            if len(labels) != info.channel_count():
                raise ValueError("--channel-order must name every stream channel, including AUX")
        indices = channel_indices(labels)
        rate = info.nominal_srate()
        if rate <= 0 or not math.isfinite(rate):
            raise ValueError("Stream must advertise a positive, finite nominal sample rate")
        return inlet, indices, rate
    except Exception:
        inlet.close_stream()
        raise


def read_sample(inlet, indices):
    sample, timestamp = inlet.pull_sample(timeout=3.0)
    if sample is None:
        raise RuntimeError("No EEG sample for three seconds; stopped instead of using stale data")
    if len(sample) <= max(indices):
        raise ValueError("EEG sample has fewer channels than its metadata")
    values = [float(sample[index]) for index in indices]
    if not np.isfinite(values).all() or not math.isfinite(timestamp):
        raise ValueError("Non-finite EEG sample or timestamp")
    return values, timestamp
