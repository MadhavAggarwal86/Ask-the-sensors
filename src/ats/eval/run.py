"""Ask the generated question set against a loaded recording and score every answer."""
import time
import zlib

from . import metrics as M
from .questions import generate


def ask_generated(qa, rid: str, gt, duration: float, has_gyro: bool) -> list[dict]:
    """qa must already hold the recording (SensorQA.load or load_features). The question set is
    seeded by the recording id, so every configuration answers exactly the same questions."""
    records = []
    for item in generate(gt, duration, has_gyro, seed=zlib.crc32(rid.encode())):
        t = time.perf_counter()
        pred = qa.ask(item.question)
        records.append({
            "rec": rid, "qtype": item.qtype, "question": item.question,
            "pred": pred, "ref": item.reference, "latency_ms": 1000 * (time.perf_counter() - t),
            "correct": M.answer_correct(item.qtype, pred, item.reference),
            "grounded": M.grounded_correct(item.qtype, pred, item.reference),
        })
    return records
