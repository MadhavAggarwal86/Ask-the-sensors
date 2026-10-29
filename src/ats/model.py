"""Recognition layer: a light classifier over engineered window features.

HistGradientBoosting is the baseline because it is small, fast on a CPU, handles missing
gyroscope features natively, and supports class weighting for the heavy sedentary imbalance.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config as C


class ActivityModel:
    def __init__(self, clf=None, feature_names: list[str] | None = None):
        self.clf = clf
        self.feature_names = feature_names or []

    @classmethod
    def new(cls, **kw) -> "ActivityModel":
        params = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                      class_weight="balanced", early_stopping=True, validation_fraction=0.1,
                      random_state=0)
        params.update(kw)
        return cls(HistGradientBoostingClassifier(**params))

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> "ActivityModel":
        """y holds class indices into config.CLASSES."""
        self.feature_names = feature_names
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray, names: list[str] | None = None) -> np.ndarray:
        """Probabilities over all of config.CLASSES, even ones missing from training.

        If the columns of X are named, the ones this model was trained on are selected, so a model
        trained on a feature subset (e.g. accelerometer only) takes the full feature matrix.
        """
        if names is not None and list(names) != list(self.feature_names):
            X = X[:, [list(names).index(n) for n in self.feature_names]]
        out = np.zeros((len(X), len(C.CLASSES)))
        if len(X):
            out[:, self.clf.classes_] = self.clf.predict_proba(X)
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "feature_names": self.feature_names, "classes": C.CLASSES}, path)

    @classmethod
    def load(cls, path: Path) -> "ActivityModel":
        d = joblib.load(path)
        if d["classes"] != C.CLASSES:
            raise ValueError("model was trained with a different class list")
        return cls(d["clf"], d["feature_names"])
