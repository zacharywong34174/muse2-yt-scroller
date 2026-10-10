"""The prototype's five time-domain features for each of four EEG channels."""

import antropy as ent
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

CHANNELS = ("AF7", "AF8", "TP9", "TP10")
STATISTICS = ("Means", "Stdevs", "Entropies", "Kurtoses", "Skewnesses")
FEATURE_NAMES = tuple(f"{stat}_{channel}" for stat in STATISTICS for channel in CHANNELS)
WINDOW_SAMPLES = 100
SLIDE_SAMPLES = 20


class FeatureExtract:
    """Preserves the original mean/std/entropy/kurtosis/skewness definitions.

    The original MNE RawArray round-trip passed its samples unchanged to Antropy;
    using that array directly removes an unnecessary dependency. No EEG filtering,
    normalization, or resampling is performed here.
    """

    def __init__(self, filepath=None, df=None):
        if filepath is not None:
            df = pd.read_csv(filepath)
        if df is None or not set(CHANNELS).issubset(df.columns):
            raise ValueError(f"Input must contain columns {', '.join(CHANNELS)}")
        self.df = df.loc[:, list(CHANNELS)].astype(float)
        if len(self.df) < 4 or not np.isfinite(self.df.to_numpy()).all():
            raise ValueError("Need at least four finite samples per channel")
        if (self.df.std() == 0).any():
            raise ValueError("Flat channels have undefined shape features; check signal contact")

    def eeg_mean(self):
        return [self.df[name].mean() for name in self.df.columns]

    def eeg_std(self):
        return [self.df[name].std() for name in self.df.columns]

    def eeg_entropy(self):
        return [ent.app_entropy(self.df[name].to_numpy(), order=2) for name in self.df.columns]

    def eeg_kurtosis(self):
        return [kurtosis(self.df[name], fisher=True, bias=False) for name in self.df.columns]

    def eeg_skewness(self):
        return [skew(self.df[name]) for name in self.df.columns]

    def frame(self):
        values = (self.eeg_mean() + self.eeg_std() + self.eeg_entropy()
                  + self.eeg_kurtosis() + self.eeg_skewness())
        if not np.isfinite(values).all():
            raise ValueError("Non-finite features; inspect the input signal")
        return pd.DataFrame([values], columns=FEATURE_NAMES)
