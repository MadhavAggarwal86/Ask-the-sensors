"""Model compression for the efficiency study: pruning, knowledge distillation, 8-bit quantisation.

Every compressed model exposes classes_ and predict_proba(X) like a scikit-learn classifier, so it
drops into ActivityModel, and therefore into the whole QA pipeline, unchanged.
"""
import copy

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

from . import config as C

K = len(C.CLASSES)


def prune_boosting(clf, n_rounds: int):
    """Keep the first n boosting rounds of a fitted HistGradientBoostingClassifier.

    Boosting adds rounds sequentially, so this is the model that training with max_iter=n_rounds
    would have produced (early stopping never triggered for the full model).
    """
    out = copy.deepcopy(clf)
    out._predictors = out._predictors[:n_rounds]
    return out


class TreeStudent:
    """A single regression tree distilled from the teacher's class probabilities (soft labels).

    Leaf values are averages of probability vectors, so the output is again a distribution and
    the HMM decoder can use it as before. Missing gyro values are routed natively by the tree.
    """

    def __init__(self, max_depth: int):
        self.tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=50, random_state=0)
        self.classes_ = np.arange(K)

    def fit(self, X, teacher_probs):
        self.tree.fit(X, teacher_probs)
        return self

    def predict_proba(self, X):
        p = np.clip(self.tree.predict(X), 1e-6, None)
        return p / p.sum(1, keepdims=True)


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class InputScaler:
    """Network input: signed log compression, standardisation, mean imputation, missing-gyro flag.

    Trees are blind to feature scale, a network is not: acc_mag_std alone spans three orders of
    magnitude. sign(x) * log(1 + |x| / median|x|) tames those tails before standardising.
    """

    def __init__(self, X):
        self.scale = (np.nanmedian(np.abs(X), 0) + 1e-6).astype(np.float32)
        L = self._log(X)
        self.mean = np.nanmean(L, 0).astype(np.float32)
        self.std = (np.nanstd(L, 0) + 1e-6).astype(np.float32)

    def _log(self, X):
        return np.sign(X) * np.log1p(np.abs(X) / self.scale)

    def __call__(self, X):
        miss = np.isnan(X)
        Z = (np.where(miss, self.mean, self._log(X)) - self.mean) / self.std
        return np.hstack([Z, miss.any(1, keepdims=True)]).astype(np.float32)


class MLPStudent:
    """A two-layer MLP distilled by regressing the teacher's centred log-probabilities (logit
    matching, Ba & Caruana 2014)."""

    def __init__(self, hidden=(128, 64)):
        self.hidden = hidden
        self.classes_ = np.arange(K)

    def fit(self, X, teacher_probs):
        self.prep = InputScaler(X)
        z = np.log(np.clip(teacher_probs, 1e-4, None))
        mlp = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                           max_iter=60, early_stopping=True, n_iter_no_change=5, random_state=0)
        mlp.fit(self.prep(X), z - z.mean(1, keepdims=True))
        # Keep only what inference needs (float32), not the optimiser state or training history.
        self.weights = [(W.astype(np.float32), b.astype(np.float32)) for W, b in zip(mlp.coefs_, mlp.intercepts_)]
        return self

    def logits(self, A):
        for i, (W, b) in enumerate(self.weights):
            A = A @ W + b
            if i < len(self.weights) - 1:
                A = np.maximum(A, 0)
        return A

    def predict_proba(self, X):
        return _softmax(self.logits(self.prep(X)))


class Int8MLP:
    """Post-training 8-bit quantisation of an MLPStudent.

    Weights are int8 with one scale per output unit. Activations entering each layer are 8-bit with
    one scale per layer, calibrated on training data: int8 for the signed network input, uint8
    after a ReLU (never negative, so all 256 levels go to the used range). Products accumulate
    exactly as in int32: the integer-valued operands are multiplied in float32, which is exact while
    |sum| < 2^24 (here at most 255 * 127 * 128 ~ 4e6), so the result equals integer arithmetic.
    """

    def __init__(self, student: MLPStudent, X_calib: np.ndarray):
        self.prep = student.prep
        self.classes_ = np.arange(K)
        self.layers = []
        A = student.prep(X_calib)
        n = len(student.weights)
        for i, (W, b) in enumerate(student.weights):
            lo, hi = (0, 255) if i > 0 else (-127, 127)
            s_in = np.float32(np.percentile(np.abs(A), 99.9) / hi)    # clip rare outliers
            s_w = (np.abs(W).max(0) / 127).astype(np.float32)
            Wq = np.clip(np.round(W / s_w), -127, 127).astype(np.int8)
            bq = np.round(b / (s_in * s_w)).astype(np.int32)
            self.layers.append((s_in, lo, hi, Wq, s_w, bq))
            A = A @ W + b
            if i < n - 1:
                A = np.maximum(A, 0)

    def predict_proba(self, X):
        A = self.prep(X)
        for i, (s_in, lo, hi, Wq, s_w, bq) in enumerate(self.layers):
            q = np.clip(np.round(A / s_in), lo, hi)                       # 8-bit activations
            A = (q @ Wq.astype(np.float32) + bq) * (s_in * s_w)            # int32 accumulate, rescale
            if i < len(self.layers) - 1:
                A = np.maximum(A, 0)
        return _softmax(A)


def n_params(clf) -> int:
    """Learned numbers in a classifier: split thresholds + leaf values for trees, weights + biases +
    input-scaler statistics for the MLPs. Feature indices and tree structure are not counted."""
    if isinstance(clf, (MLPStudent, Int8MLP)):
        prep = clf.prep.scale.size + clf.prep.mean.size + clf.prep.std.size
        if isinstance(clf, MLPStudent):
            return prep + sum(W.size + b.size for W, b in clf.weights)
        return prep + sum(Wq.size + bq.size + s_w.size + 1 for _, _, _, Wq, s_w, bq in clf.layers)
    if isinstance(clf, TreeStudent):
        t = clf.tree.tree_
        leaves = int((t.children_left == -1).sum())
        return (t.node_count - leaves) + leaves * t.value.shape[1]
    total = 0                                                   # HistGradientBoostingClassifier
    for round_ in clf._predictors:
        for tree in round_:
            leaves = int(tree.nodes["is_leaf"].sum())
            total += (len(tree.nodes) - leaves) + leaves
    return total + clf._baseline_prediction.size
