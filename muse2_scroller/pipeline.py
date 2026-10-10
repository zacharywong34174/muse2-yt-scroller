"""Shared prediction path used by the synthetic demo and live application."""

from .features import FeatureExtract


def classify_window(classifier, decision, window):
    features = FeatureExtract(df=window).frame()
    label = str(classifier.predict(features)[0])
    probabilities = classifier.predict_proba(features)[0]
    action = decision.update(label, probabilities)
    return label, probabilities, action
