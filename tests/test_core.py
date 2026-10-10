import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from muse2_scroller.decision import BlinkDecision
from muse2_scroller.demo import make_training_data, run_demo
from muse2_scroller.features import CHANNELS, FEATURE_NAMES, FeatureExtract
from muse2_scroller.live import load_model, run, send_action
from muse2_scroller.prepare import prepare
from muse2_scroller.stream import channel_indices, read_sample
from muse2_scroller.train import fit_model, training_table


@pytest.fixture
def window():
    return make_training_data(per_class=1)[0][1]


@pytest.fixture
def bundle():
    samples = make_training_data(per_class=2)
    frames = pd.concat([FeatureExtract(df=w).frame() for _, w in samples], ignore_index=True)
    return fit_model(frames, [label for label, _ in samples])


def test_features_have_stable_order_and_original_statistical_conventions(window):
    actual = FeatureExtract(df=window.loc[:, list(reversed(CHANNELS))]).frame()
    assert tuple(actual.columns) == FEATURE_NAMES
    assert actual.shape == (1, 20)
    assert np.isfinite(actual.to_numpy()).all()
    np.testing.assert_allclose(actual.iloc[0, :4], window.mean())
    np.testing.assert_allclose(actual.iloc[0, 4:8], window.std(ddof=1))
    # A row index/timestamp must not become a twenty-first feature.
    np.testing.assert_allclose(actual, FeatureExtract(df=window.assign(Timestamps=range(100))).frame())


@pytest.mark.parametrize("bad", ["missing", "nan", "flat", "short"])
def test_invalid_signal_rejected(window, bad):
    if bad == "missing":
        window = window.drop(columns="AF7")
    elif bad == "nan":
        window.iloc[0, 0] = np.nan
    elif bad == "flat":
        window["AF7"] = 0
    else:
        window = window.iloc[:3]
    with pytest.raises(ValueError):
        FeatureExtract(df=window).frame()


@pytest.mark.parametrize("label,probabilities,expected", [
    ("LB", [0.8, 0.1, 0.1], "pageup"),
    ("RB", [0.1, 0.1, 0.8], "pagedown"),
])
def test_one_action_per_completed_blink(label, probabilities, expected):
    decision = BlinkDecision()
    assert [decision.update(label, probabilities) for _ in range(5)] == [None] * 5
    assert decision.update("NB", [0.1, 0.8, 0.1]) is None
    assert decision.update("NB", [0.1, 0.8, 0.1]) == expected
    assert decision.update("NB", [0.1, 0.8, 0.1]) is None


def test_short_blinks_and_low_probability_do_not_emit_keys():
    for count, probability in [(3, [0.9, 0.05, 0.05]), (5, [0.4, 0.3, 0.3])]:
        d = BlinkDecision()
        for _ in range(count):
            assert d.update("LB", probability) is None
        assert d.update("NB", [0.1, 0.8, 0.1]) is None
        assert d.update("NB", [0.1, 0.8, 0.1]) is None


def original_episode_reference(sequence):
    """Small reference transcription of the original real_ml.py event rule."""
    active, i, j, retained = False, 0, 0, []
    actions = []
    for label, p in sequence:
        action = None
        if label != "NB" and not active:
            active = True
        if active:
            if label != "NB":
                i += 1
                if i >= 4:
                    retained.append(p)
            else:
                j += 1
                if i >= 4 and j >= 2:
                    values = np.delete(np.vstack(retained), 1, axis=1)
                    row, col = np.unravel_index(values.argmax(), values.shape)
                    if values[row, col] >= 0.5:
                        action = "pageup" if col == 0 else "pagedown"
                    active, i, j, retained = False, 0, 0, []
                elif i < 4:
                    active, i, j, retained = False, 0, 0, []
        actions.append(action)
    return actions


def test_running_max_matches_original_rule_on_mixed_sequences():
    rng = np.random.default_rng(69)
    sequence = [(str(rng.choice(["LB", "NB", "RB"])), rng.dirichlet([1, 1, 1]))
                for _ in range(3000)]
    # Explicit interrupted-neutral episode and ties.
    sequence += [("LB", [0.5, 0, 0.5])] * 4 + [
        ("NB", [0, 1, 0]), ("RB", [0.1, 0.1, 0.8]), ("NB", [0, 1, 0])]
    d = BlinkDecision()
    assert [d.update(label, p) for label, p in sequence] == original_episode_reference(sequence)


def test_class_probability_order_is_taken_from_model():
    d = BlinkDecision(["RB", "LB", "NB"])
    for _ in range(4):
        d.update("LB", [0.1, 0.8, 0.1])
    d.update("NB", [0.1, 0.1, 0.8])
    assert d.update("NB", [0.1, 0.1, 0.8]) == "pageup"


@pytest.mark.parametrize("probabilities", [[1, 0], [np.nan, 0, 0], [1.1, -0.1, 0], [0, 0, 0]])
def test_invalid_probability_rejected(probabilities):
    with pytest.raises(ValueError):
        BlinkDecision().update("LB", probabilities)


def test_training_pairs_labels_with_each_file(tmp_path):
    for index, (label, window) in enumerate(make_training_data(per_class=2)):
        folder = tmp_path / label
        folder.mkdir(exist_ok=True)
        window.to_csv(folder / f"{10-index}.csv", index=False)
    features, labels = training_table(tmp_path)
    assert labels == ["LB", "LB", "NB", "NB", "RB", "RB"]
    for index, path in enumerate(sorted((tmp_path / "LB").glob("*.csv"))):
        np.testing.assert_allclose(features.iloc[index], FeatureExtract(filepath=path).frame().iloc[0])


def test_model_round_trip_and_legacy_rejection(tmp_path, bundle):
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    assert load_model(path)["sample_rate"] == 256
    joblib.dump(bundle["classifier"], path)
    with pytest.raises(ValueError, match="Incompatible model"):
        load_model(path)


def test_channel_mapping_uses_labels():
    assert channel_indices(["TP9", "AF7", "AF8", "TP10", "AUX"]) == [1, 2, 0, 3]
    with pytest.raises(ValueError):
        channel_indices(["AF7", "AF7", "TP9", "TP10"])


class FakeInlet:
    def __init__(self, rows=(), rate=256):
        self.rows = iter(rows)
        self.count = 0
        self.rate = rate
        self.closed = False

    def pull_sample(self, timeout):
        self.count += 1
        row = next(self.rows, None)
        return row, self.count / self.rate if row is not None else None

    def close_stream(self):
        self.closed = True


def test_missing_stream_samples_timeout():
    with pytest.raises(RuntimeError, match="three seconds"):
        read_sample(FakeInlet(), [0, 1, 2, 3])


def test_live_uses_100_sample_window_20_sample_slide_and_closes(monkeypatch, bundle):
    rng = np.random.default_rng(10)
    inlet = FakeInlet(rng.normal(size=(200, 4)))
    monkeypatch.setattr("muse2_scroller.live.open_stream", lambda *a: (inlet, [0, 1, 2, 3], 256))
    lengths = []

    def classify(*args):
        lengths.append(len(args[-1]))
        return "NB", [0, 1, 0], None

    monkeypatch.setattr("muse2_scroller.live.classify_window", classify)
    run(bundle, max_windows=3)
    assert lengths == [100, 100, 100]
    assert inlet.count == 160
    assert inlet.closed


def test_sample_rate_mismatch_stops_before_reading(monkeypatch, bundle):
    inlet = FakeInlet()
    monkeypatch.setattr("muse2_scroller.live.open_stream", lambda *a: (inlet, [0, 1, 2, 3], 64))
    with pytest.raises(ValueError, match="sample rate differs"):
        run(bundle, max_windows=1)
    assert inlet.count == 0 and inlet.closed


def test_stream_loss_closes_inlet(monkeypatch, bundle):
    inlet = FakeInlet()
    monkeypatch.setattr("muse2_scroller.live.open_stream", lambda *a: (inlet, [0, 1, 2, 3], 256))
    with pytest.raises(RuntimeError, match="No EEG sample"):
        run(bundle, max_windows=1)
    assert inlet.closed


def test_dry_run_never_imports_keyboard_library(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pyautogui", None)
    send_action("pageup")
    assert "DRY RUN: pageup" in capsys.readouterr().out


def test_prepare_validates_before_writing_and_refuses_overwrite(tmp_path, window):
    recording, labels, output = tmp_path / "raw.csv", tmp_path / "labels.csv", tmp_path / "windows"
    window.to_csv(recording, index=False)
    labels.write_text("label,start_sample\nLB,0\nRB,1\n")
    with pytest.raises(ValueError):
        prepare(recording, labels, output)
    assert not output.exists()
    labels.write_text("label,start_sample\nLB,0\n")
    assert prepare(recording, labels, output) == 1
    np.testing.assert_allclose(pd.read_csv(output / "LB/window_0000.csv"), window)
    with pytest.raises(ValueError, match="exists"):
        prepare(recording, labels, output)


def test_synthetic_demo_end_to_end():
    assert run_demo() == ["pageup", "pagedown"]
