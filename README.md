# Muse 2 · Blink-to-Scroll

A Python prototype that turns intentional left/right blinks, observed in a
Muse 2 EEG stream, into Page Up / Page Down commands for YouTube Shorts or another
scrollable page. Built by **Zachary Wong** as a personal signal-processing and
human–computer interaction project.

**Try it without a headset:** the included synthetic demo runs the feature
extraction → random forest → blink-event rule end to end. It uses generated toy
signals, sends no keystrokes, and does not connect to YouTube.

This is an experimental code sample, not a medical device or a validated
accessibility product. Blinks produce ocular/muscle artifacts in EEG; this project
uses those artifacts as intentional inputs. It does not decode thoughts.

## Quick start — no hardware or private data

Use **Python 3.12** (the locally tested version). Python 3.11–3.13 are declared
compatible; other versions have not been validated. Download/extract this source
or clone the repository, then open a terminal in its root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m muse2_scroller.demo
python -m pytest -q
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` in PowerShell. All subsequent `python -m ...`
commands are the same. No GPU, API key, account, or browser driver is required.

For the exact resolved dependency set, [uv](https://docs.astral.sh/uv/) users can
instead run `uv sync --locked --extra test --python 3.12`, then
`uv run --locked --extra test python -m muse2_scroller.demo`. `uv.lock` also locks
the optional `live` extra; add `--extra live` when syncing for hardware use.

Expected final demo line:

```text
PASS: pageup, pagedown. This demonstrates software flow, not real-world accuracy.
```

The demo trains a small random forest in memory on deliberately separable toy
windows and classifies a separate seeded sequence. It demonstrates the plumbing,
not physiological realism, measured latency, or performance on a person.

To exercise the CSV → model-file route as well:

```bash
python -m muse2_scroller.demo --write-training demo-output/windows
python -m muse2_scroller.train --data-dir demo-output/windows --model demo-output/toy.joblib
```

Both commands refuse to overwrite existing outputs. Never use the toy model to
control your computer from real EEG.

## How it works

```text
Muse 2 → separate Bluetooth-to-LSL bridge → labelled EEG channels
       → 100-sample window, advanced by 20 samples
       → 5 statistics × 4 channels = 20 features
       → random forest: LB / NB / RB
       → blink-event aggregation → dry-run log or Page Up / Page Down
```

The features are mean, sample standard deviation, approximate entropy (order 2),
unbiased Fisher kurtosis, and skewness. The forest uses 100 trees and a fixed
random seed. Features are ordered consistently as AF7, AF8, TP9, TP10 within each
statistic; incoming LSL samples are mapped using channel metadata.

The event rule retains the prototype's behavior: collect at least four non-neutral
predictions, retain probabilities starting at the fourth, then wait for two
neutral observations. If the largest retained LB/RB probability is ≥ 0.5,
emit one action: **LB → Page Up**, **RB → Page Down**. The two neutral observations
need not be consecutive. Short blink runs are discarded. See
[design notes](docs/DESIGN.md) for timing and tradeoffs.

## Using a Muse 2

You need a charged Muse 2, a computer with compatible Bluetooth, a working
Bluetooth-to-LSL bridge, and a model trained on your own labelled recordings.
The repository deliberately contains no personal EEG or pretrained personal model.

1. Install the optional local acquisition/control dependencies:

   ```bash
   python -m pip install -e ".[live]"
   python -c "import pylsl; print(pylsl.library_version())"
   ```

2. Set up a separate EEG stream. [MuseLSL](https://github.com/alexandrebarachant/muse-lsl)
   supports Muse 2 and documents Bluetooth setup for each OS; Windows users may
   also use its linked BlueMuse option. Install the bridge according to its own
   instructions, then keep `muselsl stream` running in a separate terminal.
   This application consumes LSL; it does not pair with the headset itself.
   If `pylsl` cannot find its native library, follow the
   [official liblsl installation/loading instructions](https://github.com/labstreaminglayer/pylsl#liblsl-loading).

3. Collect and label your own data using the
   [recording and training guide](docs/TRAINING.md). Training and live acquisition
   must use the same sample rate and correctly identified electrodes.

4. Start with a dry run. This prints predictions and intended actions only:

   ```bash
   python -m muse2_scroller.live --model models/blinks.joblib --max-windows 100
   ```

5. After checking detection in a dry run, open a safe scrollable test page. Enable
   keyboard output explicitly, then focus that page during the five-second delay:

   ```bash
   python -m muse2_scroller.live --model models/blinks.joblib --enable-control
   ```

Page keys act on the **currently focused application**, not specifically YouTube.
Page Up/Down behavior varies by browser and site; first verify the same keys
manually. macOS may require Accessibility permission for the terminal/Python
application. Linux desktop control generally requires an X11 session; Wayland
restrictions may prevent it. The demo does not require any of these permissions.

Stop with **Ctrl+C in the terminal**. With control enabled, moving the pointer to a
corner of the primary monitor triggers PyAutoGUI's
[fail-safe on the next keyboard action](https://pyautogui.readthedocs.io/en/latest/#fail-safes).
It is not an independent emergency-stop listener. A stream timeout or invalid
signal stops the application and closes its LSL inlet.

## Limitations and responsible use

- No live-device accuracy, false-positive rate, cross-person generalization, or
  end-to-end latency was measured for this release. Device acquisition, Bluetooth,
  intentional-blink detection, OS key delivery, and YouTube navigation remain
  untested here. See the [validation record](docs/VALIDATION.md).
- Motion, poor electrode contact, facial muscle activity, ordinary blinks, and
  fatigue may cause false actions. Use it seated in a safe setting, take breaks,
  and stop if repeated blinking becomes uncomfortable. Do not use it for
  safety-critical control or as a substitute for an established assistive device.
- A classifier trained on one person/session may fail on another. Overlapping
  training and evaluation windows can inflate apparent accuracy; hold out entire
  recording sessions. Class probabilities are not calibrated confidence measures.
- Joblib models can execute Python code when loaded. Only load models you created
  or otherwise trust. The original personal model is not distributed.
- Keep recordings, labels, device identifiers, and videos private unless everyone
  involved has agreed to their release. LSL can expose streams to the local network;
  use a trusted network and stop the bridge after use.

## A two-minute portfolio walkthrough

1. Run the synthetic demo and show the two logged actions.
2. Open [features.py](muse2_scroller/features.py) and
   [decision.py](muse2_scroller/decision.py) to explain the signal features and
   event aggregation; then show a passing test run.
3. If recording a hardware demo later, use a page and media you own, show the
   headset and deliberate blink, and label the recording as a live demonstration.
   Hide account details and recording paths. Do not include third-party YouTube
   videos, music, or private EEG in a public demo without appropriate permission.

## Code map and provenance

| File | Responsibility |
| --- | --- |
| `features.py` | Original feature definitions, with explicit schema validation |
| `decision.py` | Original blink rule, isolated from hardware for testing |
| `pipeline.py`, `live.py` | Shared inference and bounded streaming loop |
| `stream.py`, `record.py` | LSL channel mapping and local CSV recording |
| `prepare.py`, `train.py` | Labelled-window preparation and deterministic training |
| `demo.py`, `tests/` | Generated signals and hardware-independent checks |

The original working folder was preserved unchanged. This public preparation
curates the working inference/training path and adds installation metadata,
portable paths, documentation, validation, and safe defaults. These release
preparation changes were made with AI assistance; the original prototype and
these later changes should not be described as the same authorship exercise.
See [provenance and changes](docs/PROVENANCE.md) and the
[publication audit](docs/AUDIT.md). Exploratory scripts, unreviewed tutorial-like
fragments, cached bytecode, recordings, plots, and a personal fitted model are
excluded from this distribution.

## License

Project code is provided under the [MIT License](LICENSE). Dependencies retain
their own licenses and are installed separately; none are vendored. Muse and
YouTube names describe compatibility only. This is an independent project with
no affiliation or endorsement from their owners.
