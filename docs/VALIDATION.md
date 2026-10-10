# Validation record

Date: 2026-10-10. Local platform: macOS on Apple Silicon, Python 3.12.13.

## Completed

- Fresh editable installation of the core, test, and optional live dependencies.
- A second installation from `uv.lock`; lock consistency checked successfully.
- **24 automated tests passed**. Coverage includes feature ordering and statistical
  conventions; non-finite/flat/missing inputs; blink suppression and aggregation;
  3,000 randomized events compared against the original state-machine rule;
  reordered model classes; paired features/labels; model save/reload; legacy-model
  rejection; electrode mapping; dry-run keyboard isolation; 100/20 windowing;
  sample-rate mismatch; stream timeout/cleanup; annotation bounds; and the demo.
- The deterministic synthetic demo produced exactly `pageup`, then `pagedown`.
- Native **liblsl 118** loaded. A separate synthetic LSL outlet published five
  labelled channels in TP9, AF7, AF8, TP10, AUX order at 256 Hz. The application
  resolved the explicitly selected test source, ran three dry-run predictions,
  recorded exactly 512 samples over a two-second sample-count target, prepared
  three labelled windows, and trained a model from those files. No real EEG or
  keyboard control was used in this check.
- Every Python source/test file parsed successfully. Automated secret scanning
  reported no candidates in the release, and all 127 original files retained
  their original SHA-256 hashes.

Representative direct dependency versions: NumPy 2.5.3, pandas 2.3.3,
SciPy 1.18.1, scikit-learn 1.9.1, Antropy 0.2.2, joblib 1.6.0,
pylsl 1.18.5, PyAutoGUI 0.9.54, and pytest 9.1.1. See `uv.lock` for all
dependencies, hashes, and Python/platform-specific resolution branches.

## Not established by these checks

No physical Muse 2 was used. Bluetooth pairing, electrode contact, participant
calibration, real blink classification, false-action rate, perceived latency,
actual OS keystrokes, and YouTube navigation have not been validated for this
release. The original model was not loaded. These checks establish software
behavior on generated signals only, not clinical, physiological, or accessibility
effectiveness. Windows/Linux/macOS permission flows were not exercised.

A GitHub Actions workflow is included for Python 3.11, 3.12, and 3.13 on Linux.
The local results above do not imply that a remote CI run has completed; consult
the repository's Actions tab for its current status.
