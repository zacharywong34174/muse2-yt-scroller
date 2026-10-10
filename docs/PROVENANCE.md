# Provenance and release preparation

The input was a personal working folder, not a Git repository. Its ten Python
files contained 1,550 newline-counted lines, including comments, inactive code,
and exploratory programs. This is not a count of independently authored executable
lines. No earlier commit history was available to reconstruct.

The source folder, recordings, annotations, plots, bytecode, and fitted model
were left in place. A private SHA-256 inventory was taken before editing the
separate release copy and checked afterward. Raw originals were never committed
to this repository; no sensitive baseline commit is hidden in its history.

| Original component | Curated implementation |
| --- | --- |
| `functional/feature_extraction.py` | `features.py`: retains the five feature definitions; removes unused MNE wrapping and adds validation/order |
| `functional/real_ml.py` | `decision.py`, `pipeline.py`, `live.py`: retains 100/20 windowing and blink aggregation; adds dry run, errors, and cleanup |
| `functional/train_ml.py` | `train.py`: retains random forest settings; pairs features with labels and saves sampling metadata |
| `functional/brain_data_L_extractor.py` | `prepare.py`: replaces spreadsheet row conventions and fixed paths with explicit sample indices |
| Keyboard helper | `live.send_action`: retains Page Up/Down mapping; removes unused Selenium/browser creation |
| Plotting recorder | `record.py`: small CSV recorder that saves every sample; plotting/UI experiments remain in the original folder |

The extracted decision rule uses a bounded running maximum in place of a growing
probability list. Tests compare it against the original state-machine behavior,
including nonconsecutive neutral observations. Channel mapping and consistent
recording rates are deliberate corrections; live hardware equivalence is untested.

Documentation, packaging, test coverage, the synthetic demo, command-line
interfaces, and release refactoring were prepared with AI assistance. Do not
represent all lines of this curated release as independently handwritten, or
attribute the original prototype's informal device observations to this release's
tests. The repository is intended to expose the implementation and its limits.

Unused experiments and inactive tutorial-like plotting/browser snippets were not
included because they do not support the working path and their provenance was
not established. No third-party library source or media is bundled. The original
folder contained no license file or explicit third-party copyright notice; that
absence alone does not prove ownership. MIT applies to project code that the
author has the right to license; any externally sourced code must retain its
applicable notice before being added.
