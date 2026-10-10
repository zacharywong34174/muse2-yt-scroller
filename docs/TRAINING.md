# Recording and training your own model

The synthetic quick start is sufficient for reviewing the software. Live use
requires fresh, private training data from the person using it. This guide is a
software workflow, not a validated data-collection protocol.

## Record

Start an EEG LSL bridge as described in the README, then:

```bash
python -m muse2_scroller.record --seconds 60 --output data/session01.csv
```

This writes `Timestamps,AF7,AF8,TP9,TP10`, retaining each received sample. It refuses
to overwrite an existing recording. The terminal prints the nominal sample rate.
Record on a trusted network and stop the bridge when finished. Abort with Ctrl+C;
received samples are retained locally. A stopped/failed recording can be partial.

If more than one EEG stream exists, pass `--source-id` using the identifier from
your bridge. If channel labels are absent, pass `--channel-order` matching the
bridge's actual ordering, including any auxiliary channel; do not guess. The same
options are available on `muse2_scroller.live`.

## Annotate

Inspect your recording and label windows as LB (intentional left blink), NB (no
intentional blink), and RB (intentional right blink). Create a private CSV such as:

```csv
label,start_sample
NB,0
LB,300
RB,700
```

These are illustrative indices, not labels for any real recording. `start_sample`
is zero-based and excludes the CSV header; each annotation selects exactly 100
rows, ending just before `start_sample + 100`. The software validates all
annotations before creating output. Include diverse neutral/movement examples,
avoid overlapping windows across evaluation splits, and collect separate sessions
for evaluation. Three example rows are not a sufficient real training dataset.

```bash
python -m muse2_scroller.prepare --recording data/session01.csv --annotations data/session01-labels.csv --output data/windows01
```

The result is `data/windows01/LB/*.csv`, `NB/*.csv`, and `RB/*.csv`. All three
labels must be present before training. Each CSV must contain 100 rows of finite,
non-flat AF7, AF8, TP9, TP10 samples, in the same units used for live acquisition.
Prepare one session at a time; keep a separate session completely out of training.
The command refuses an existing output directory to prevent mixing sessions.

## Train, then inspect predictions

For a recording at 256 Hz:

```bash
python -m muse2_scroller.train --data-dir data/windows01 --sample-rate 256 --model models/blinks.joblib
python -m muse2_scroller.live --model models/blinks.joblib --max-windows 100
```

The training command creates features and labels from each file together in
sorted order. It trains on all supplied windows with 100 trees and seed 69,
then saves a local model bundle. It reports no cross-validation accuracy:
randomly splitting correlated windows is not evidence of generalization.
Only enable keyboard control after reviewing behavior against intentionally
performed actions in a separate session.

Only load models you created or trust: joblib deserialization can execute code.
Package versions used to train and load a model should match. Existing prototype
models lack rate/schema metadata and are rejected; retraining is intentional.
