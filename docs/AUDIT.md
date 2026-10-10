# Publication audit

Reviewed on 2026-10-10. Scope: the supplied local folder and the curated release
tree/history. This is a practical publication check, not proof of the absence of
every possible secret or a legal opinion about ownership.

| Area | Finding and release treatment |
| --- | --- |
| Git history | No source repository or history existed. This repository starts with reviewed files only. |
| Credentials | Pattern review of source/text files found no API keys, tokens, passwords, private keys, or device MAC addresses. `detect-secrets` 1.5.0 found zero candidates in the prepared release tree; the public package lock was separately checked for private URLs/paths. |
| Personal data | 106 CSV files, including raw EEG, annotations, and derived features, are excluded. No personal signal values are embedded in fixtures. |
| Fitted model | The personal `.joblib` artifact is excluded and was not deserialized. Models must be trained locally. |
| Absolute paths | Hard-coded personal Windows paths were removed by using explicit command-line paths. No home-directory paths or private endpoints appear in released files. |
| Media | Five PNG plots are excluded. No source videos/audio were present, and no YouTube media is redistributed. |
| Large/generated files | The approximately 15.4 MB raw recording, bytecode caches, plots, and model are excluded. All published files are text; the dependency lock is the largest. |
| Licensing | No original license or explicit third-party copyright notice was found. Inactive tutorial-like plotting/browser fragments and unused experiments with unestablished provenance were omitted. MIT is included for project code; dependency licenses remain separate. |
| Git identity | Local release commits use a GitHub no-reply address rather than a personal email address. |
| Future additions | `.gitignore` excludes credentials, recordings, models, media, environments, and build output. Ignoring files is not a substitute for reviewing staged changes, especially when using GitHub's web uploader. |

The review does not establish the origin of every line of the historical folder.
Do not add externally sourced code or media without establishing its license and
retaining required notices. No dependency source is vendored into this repository.

## Dependencies

The core uses NumPy, pandas, SciPy, scikit-learn, Antropy, and joblib. These projects
publish permissive licensing information in their distribution metadata; NumPy
and SciPy distributions also carry notices for bundled components. Optional
hardware support uses pylsl (MIT metadata) and PyAutoGUI (BSD metadata).
The separate MuseLSL bridge has its own BSD-3-Clause license and is not bundled.
Consult installed package notices and each upstream project for the full terms.

Unused Selenium/webdriver-manager, PyQt5, MNE, and experimental ICA/visualization
dependencies are not needed by the curated runnable path. Removing them reduces
installation requirements and avoids redistributing optional GUI components or
unreviewed snippets. `uv.lock` records resolved versions and distribution hashes.

## Data handling

The synthetic generator uses fixed random seeds and mathematical pulses; it does
not transform or sample the private recordings. Test artifacts remain outside
the release. The live application logs labels/probabilities, does not save EEG,
and only sends keys when explicitly enabled. The separate recorder writes data
to the local path requested by its operator. LSL stream exposure and browser
behavior still depend on the operator's local environment.
