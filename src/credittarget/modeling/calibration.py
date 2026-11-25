"""Version-safe PlattCalibratedClassifier wrapping BaseEstimator with 1-D logistic recalibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


class PlattCalibratedClassifier(BaseEstimator, ClassifierMixin):
    """Post-hoc Platt calibrator fitting a 1-D logistic model on log-odds of a pre-trained base model.

    Calibrated strictly on Month 9 validation data, fully picklable, and free of sklearn 1.6+ deprecation warnings.
    """

    def __init__(self, base_estimator: Any) -> None:
        self.base_estimator = base_estimator
        self.calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        self.classes_ = np.array([0, 1])

    def fit(self, X_val: Any, y_val: Any) -> PlattCalibratedClassifier:
        """Fits 1-D logistic calibrator on the log-odds of base estimator predictions."""
        # Get raw probability of positive class
        raw_proba = self.base_estimator.predict_proba(X_val)[:, 1]
        clipped_proba = np.clip(raw_proba, 1e-6, 1.0 - 1e-6)
        log_odds = np.log(clipped_proba / (1.0 - clipped_proba)).reshape(-1, 1)

        self.calibrator.fit(log_odds, y_val)
        return self

    def predict_proba(self, X: Any) -> np.ndarray[Any, Any]:
        """Outputs calibrated [P(y=0), P(y=1)] probabilities."""
        raw_proba = self.base_estimator.predict_proba(X)[:, 1]
        clipped_proba = np.clip(raw_proba, 1e-6, 1.0 - 1e-6)
        log_odds = np.log(clipped_proba / (1.0 - clipped_proba)).reshape(-1, 1)

        calibrated_p1 = self.calibrator.predict_proba(log_odds)[:, 1]
        calibrated_p0 = 1.0 - calibrated_p1
        return np.column_stack([calibrated_p0, calibrated_p1])

    def predict(self, X: Any) -> np.ndarray[Any, Any]:
        """Outputs binary class prediction using standard 0.5 decision threshold."""
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)
