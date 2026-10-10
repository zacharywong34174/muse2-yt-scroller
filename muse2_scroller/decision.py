"""The original blink-event rule, separated from acquisition and keyboard IO."""

import numpy as np


class BlinkDecision:
    """Aggregate a blink episode into at most one Page Up / Page Down action.

    Retains the prototype's semantics: four non-NB predictions are required,
    probabilities are retained starting at the fourth, and two NB observations
    end the episode. Those NB observations need not be consecutive. An NB before
    four blink predictions cancels the episode. The largest retained LB/RB
    probability must be at least 0.5. This is a heuristic, not a calibrated score.
    """

    def __init__(self, classes=("LB", "NB", "RB")):
        self.classes = tuple(classes)
        if len(self.classes) != 3 or set(self.classes) != {"LB", "NB", "RB"}:
            raise ValueError("Model must classify LB, NB, and RB")
        self.reset()

    def reset(self):
        self.blink_count = 0
        self.neutral_count = 0
        # Only the largest probability matters; avoid an unbounded list.
        self.best_probability = -1.0
        self.best_action = None

    def update(self, label, probabilities):
        p = np.asarray(probabilities, dtype=float)
        if (label not in self.classes or p.shape != (3,) or not np.isfinite(p).all()
                or (p < 0).any() or (p > 1).any() or not np.isclose(p.sum(), 1)):
            raise ValueError("Expected a known class and three valid class probabilities")
        if label != "NB":
            self.blink_count += 1
            if self.blink_count >= 4:
                for blink, action in (("LB", "pageup"), ("RB", "pagedown")):
                    probability = p[self.classes.index(blink)]
                    if probability > self.best_probability:
                        self.best_probability = probability
                        self.best_action = action
            return None
        if self.blink_count < 4:
            self.reset()
            return None
        self.neutral_count += 1
        if self.neutral_count < 2:
            return None
        action = self.best_action if self.best_probability >= 0.5 else None
        self.reset()
        return action
