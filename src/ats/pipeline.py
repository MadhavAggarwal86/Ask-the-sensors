"""End-to-end system: recording + question -> grounded answer."""
from pathlib import Path

from . import config as C
from .features import extract
from .model import ActivityModel
from .preprocess import Recording, make_windows
from .qa import operations
from .qa.parser import parse
from .timeline import Timeline


class SensorQA:
    def __init__(self, model: ActivityModel, parser=None, smoothing: str = C.SMOOTHING):
        self.model = model
        self.parser = parser or parse   # swap in qa.llm.LLMParser(...) for the SLM front end
        self.smoothing = smoothing
        self.timeline: Timeline | None = None
        self.duration, self.t0 = 0.0, None

    @classmethod
    def from_path(cls, model_path: Path = C.MODELS_DIR / "hgb_full.joblib", **kw) -> "SensorQA":
        return cls(ActivityModel.load(model_path), **kw)

    def load(self, rec: Recording) -> Timeline:
        """Run preprocessing, recognition and aggregation once per recording."""
        starts, wins = make_windows(rec)
        feats, names = extract(wins) if len(wins) else (None, [])
        return self.load_features(starts, feats, names, rec.has_gyro, rec.duration, rec.t0)

    def load_features(self, starts, feats, names, has_gyro: bool, duration: float, t0=None) -> Timeline:
        """load() from precomputed window features (the efficiency benchmark reuses one extraction)."""
        self.duration, self.t0 = duration, t0
        if feats is None or not len(feats):
            self.timeline = Timeline([], has_gyro=has_gyro, duration=duration)
        else:
            probs = self.model.predict_proba(feats, names)
            self.timeline = Timeline.from_predictions(starts, probs, feats, names, has_gyro, duration,
                                                      self.smoothing)
        return self.timeline

    def ask(self, question: str) -> operations.Answer:
        if self.timeline is None:
            raise RuntimeError("call load(recording) first")
        intent = self.parser(question, duration=self.duration, t0=self.t0)
        return operations.run(intent, self.timeline)
