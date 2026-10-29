"""Grounded explanations built only from measured feature values.

Nothing here is free text from a model: each sentence is a template filled with numbers that
were computed from the cited windows, so the explanation cannot drift from the evidence.
"""
import numpy as np

from .. import config as C

WHY = {
    "lying_down": "very little motion with a device orientation that differs from the upright posture, "
                  "consistent with lying down rather than sitting or standing",
    "sitting": "very little motion with a stable, non-upright orientation, consistent with sitting",
    "standing_in_place": "very little motion while the orientation stays close to upright, "
                         "consistent with standing still",
    "standing_moving": "irregular, low-amplitude movement without a steady gait rhythm, consistent with "
                       "standing and shifting or moving around in place",
    "walking": "a regular gait rhythm at walking cadence with moderate heel-strike impacts",
    "running": "high-amplitude impacts at a faster cadence than walking, with large rotations",
    "bicycling": "smooth cyclic motion with little impact energy and sustained periodic rotation, "
                 "consistent with pedalling rather than stepping",
}


def describe_signal(s: dict[str, float], has_gyro: bool) -> str:
    """One sentence summarising the measured signal properties of a set of windows."""
    if not s:
        return "no windows available to describe"
    parts = [f"accelerometer magnitude variability of {s['acc_mag_std']:.3f} g"]
    if s["acc_mag_std"] > 0.04 and s["acc_periodicity"] > 0.25:
        f = s["acc_dom_freq_hz"]
        parts.append(f"a regular rhythm at {f:.2f} Hz (about {f * 60:.0f} cycles/min, "
                     f"periodicity {s['acc_periodicity']:.2f})")
    else:
        parts.append(f"no dominant rhythm (periodicity {s['acc_periodicity']:.2f})")
    if has_gyro and np.isfinite(s.get("gyro_mag_mean", np.nan)):
        parts.append(f"gyroscope rotation averaging {s['gyro_mag_mean']:.2f} rad/s")
    tilts = {a: s.get(f"tilt_{a}_deg", np.nan) for a in "xyz"}
    if all(np.isfinite(v) for v in tilts.values()):
        axis = min(tilts, key=lambda a: min(tilts[a], 180 - tilts[a]))
        ang = min(tilts[axis], 180 - tilts[axis])
        parts.append(f"gravity within {ang:.0f} degrees of the device {axis.upper()}-axis")
    return ", ".join(parts)


def explain_activity(activity: str, stats: dict, contrast: dict, has_gyro: bool) -> str:
    text = f"The cited windows show {describe_signal(stats, has_gyro)}: {WHY.get(activity, 'see measured values')}."
    if stats and contrast and "acc_mag_std" in contrast:
        text += (f" For contrast, the rest of the recording averages {contrast['acc_mag_std']:.3f} g "
                 f"variability.")
    return text


def modality(has_gyro: bool) -> tuple[str, str]:
    return ("Accelerometer, Gyroscope", "All") if has_gyro else ("Accelerometer", "Acc X/Y/Z")
