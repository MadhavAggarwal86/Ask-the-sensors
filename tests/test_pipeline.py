"""Fast checks of every layer on synthetic data. Run with: pytest -q"""
import numpy as np
import pytest

from ats import config as C
from ats.eval.metrics import interval_iou
from ats.features import extract
from ats.model import ActivityModel
from ats.pipeline import SensorQA
from ats.preprocess import make_windows
from ats.qa.parser import parse
from ats.synthetic import make_recording

SCRIPT = [("sitting", 240), ("walking", 180), ("running", 120), ("standing_in_place", 90),
          ("bicycling", 180), ("lying_down", 700), ("walking", 120)]


@pytest.fixture(scope="module")
def qa():
    X, y = [], []
    for seed in range(4):
        rec, gt = make_recording(SCRIPT[::-1] if seed % 2 else SCRIPT, seed=seed)
        starts, wins = make_windows(rec)
        F, names = extract(wins)
        centres = starts + C.WIN_S / 2
        lab = np.array([next((C.CLASSES.index(a) for s, e, a in gt if s <= c < e), -1) for c in centres])
        X.append(F[lab >= 0]); y.append(lab[lab >= 0])
    X.append(make_windows(make_recording([("standing_moving", 300)], seed=9)[0])[1])
    model = ActivityModel.new(max_iter=100)
    Fsm, _ = extract(X.pop())
    model.fit(np.vstack(X + [Fsm]), np.concatenate(y + [np.full(len(Fsm), C.CLASSES.index("standing_moving"))]), names)
    system = SensorQA(model)
    system.load(make_recording(SCRIPT, seed=42)[0])
    return system


@pytest.mark.parametrize("q,op,acts", [
    ("Is the user running?", "verify", [["running"]]),
    ("How long was the user walking?", "duration", [["walking"]]),
    ("Did the user spend more time walking or running?", "compare", [["walking"], ["running"]]),
    ("What activity is the user performing?", "identify", []),
    ("Did the user lie down for a prolonged period?", "prolonged", [["lying_down"]]),
    ("Was the user using a wheeled or pedal-based mode of movement?", "verify", [["bicycling"]]),
    ("Did the user begin running at any point, and if so, when?", "onset", [["running"]]),
    ("How many times did the user walk?", "count", [["walking"]]),
    ("Is the recording consistent with the user exercising?", "open", []),
])
def test_parser(q, op, acts):
    it = parse(q)
    assert it.op == op
    assert it.activities == acts


def test_parser_time_range():
    assert parse("What was the user doing between 100 and 200 seconds?").t_range == (100, 200)
    assert parse("Was the user sitting in the first 5 minutes?").t_range == (0, 300)


def test_windows_skip_gaps():
    rec, _ = make_recording([("sitting", 300)], seed=0, bursty=True)
    starts, wins = make_windows(rec)
    assert len(starts) > 0
    assert not any(20 < s % 60 < 55 for s in starts)   # no windows inside the 40 s gaps
    assert np.allclose(np.diff(starts)[np.diff(starts) < 10], C.HOP_S)


def test_end_to_end(qa):
    a = qa.ask("How long was the user walking?")
    assert abs(a.value - 300) <= 30
    assert a.modality == "Accelerometer, Gyroscope"
    assert a.intervals and a.explanation != "N/A"
    a = qa.ask("Did the user begin running at any point, and if so, when?")
    assert interval_iou(a.intervals, [(420, 540)]) > 0.7
    assert qa.ask("Did the user lie down for a prolonged period?").value is True
    assert qa.ask("Did the user spend more time walking or running?").answer == "Walking"
    assert qa.ask("What was the user doing between 600 and 650 seconds?").value == "standing_in_place"


def test_no_gyro_reports_accelerometer_only(qa):
    rec, _ = make_recording(SCRIPT, seed=7, drop_gyro=True)
    qa.load(rec)
    a = qa.ask("How long was the user walking?")
    assert a.modality == "Accelerometer" and a.channels == "Acc X/Y/Z"


def test_viterbi_removes_flicker_keeps_real_change():
    from ats.timeline import viterbi
    starts = np.arange(40) * C.HOP_S
    p = np.full((40, 7), 0.02)
    p[:, 1] = 0.88
    p[10, 1], p[10, 4] = 0.2, 0.7          # one misclassified window
    p[25:, 1], p[25:, 4] = 0.05, 0.85      # a genuine change to walking
    path = viterbi(starts, p, tau_s=600)
    assert (path[:25] == 1).all() and (path[25:] == 4).all()
    # A data gap too long to bridge splits decoding, so the second part stands alone.
    starts[20:] += 1000
    assert viterbi(starts, p, tau_s=600)[20] == 1


def test_compressed_models_follow_teacher(qa):
    from ats.compress import Int8MLP, MLPStudent, TreeStudent, prune_boosting
    rec, _ = make_recording(SCRIPT, seed=7)
    X, names = extract(make_windows(rec)[1])
    teacher = qa.model.predict_proba(X)
    pruned = ActivityModel(prune_boosting(qa.model.clf, 10), names)
    assert pruned.clf.n_iter_ == 10 and pruned.predict_proba(X).shape == teacher.shape
    tree = TreeStudent(6).fit(X, teacher)
    mlp = MLPStudent(hidden=(32,)).fit(X, teacher)
    q = Int8MLP(mlp, X)
    assert all(l[3].dtype == np.int8 for l in q.layers)
    for student in (tree, mlp, q):
        p = student.predict_proba(X)
        assert np.allclose(p.sum(1), 1) and (p.argmax(1) == teacher.argmax(1)).mean() > 0.7
    # Missing gyroscope must not break any student.
    Xn = X.copy()
    Xn[:, [i for i, n in enumerate(names) if n.startswith("gyro")]] = np.nan
    assert all(np.isfinite(s.predict_proba(Xn)).all() for s in (tree, mlp, q))
