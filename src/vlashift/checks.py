"""Checks on what the hooks did and on the match between LIBERO and the instruction table."""

import json
import math
from pathlib import Path

from vlashift.config import Arm, RuntimeConfig
from vlashift.suites import instructions

MIN_EEF_SHIFT = 1e-3
IMAGE_SHIFTS = ("bright", "noise", "occl")


def read_applied(directory: Path) -> dict:
    return json.loads((Path(directory) / "vlashift_applied.json").read_text(encoding="utf-8"))


def applied_problems(arm: Arm, applied: dict, baseline: dict | None = None) -> list[str]:
    """What is wrong with a run, judged from the record the hooks wrote; empty means fine."""
    counts = applied["counts"]
    expected = RuntimeConfig.from_environ(arm.env())
    problems = []
    for field in ("instruction", "brightness", "noise_std", "occlusion", "joint_std"):
        got, want = applied["config"][field], getattr(expected, field)
        if got != want:
            problems.append(f"setting {field} was {got!r}, the arm {arm.name} needs {want!r}")
    changed = counts["instructions_changed"]
    if arm.instruction == "keep" and changed:
        problems.append(f"the instruction was changed {changed} times in an arm that keeps it")
    if arm.instruction != "keep" and not changed:
        problems.append("the instruction was never changed")
    perturbed = counts["images_perturbed"]
    if arm.shift in IMAGE_SHIFTS and not perturbed:
        problems.append("no image was perturbed")
    if arm.shift not in IMAGE_SHIFTS and perturbed:
        problems.append(f"{perturbed} images were perturbed in an arm without an image shift")
    states = counts["init_states_perturbed"]
    if arm.shift == "joint" and not states:
        problems.append("no init state was perturbed (do envs run in worker processes?)")
    if arm.shift != "joint" and states:
        problems.append(f"{states} init states were perturbed in an arm without joint noise")
    if arm.shift == "joint" and baseline is not None:
        problems += _eef_problems(applied, baseline)
    return problems


def _eef_problems(applied: dict, baseline: dict) -> list[str]:
    if not applied["first_states"] or not baseline["first_states"]:
        return ["no first state was recorded, cannot compare the start pose with the baseline"]
    gap = math.dist(applied["first_states"][0][:3], baseline["first_states"][0][:3])
    if gap < MIN_EEF_SHIFT:
        return [f"end-effector start moved by {gap:.5f} m, joint noise had no visible effect"]
    return []


def description_problems(directory: Path) -> list[str]:
    """Compare task_descriptions.json with the table; a task name ending in the text passes."""
    path = Path(directory) / "task_descriptions.json"
    if not path.is_file():
        return [f"{path.name} is missing"]
    problems = []
    for key, text in json.loads(path.read_text(encoding="utf-8")).items():
        suite, _, index = key.rpartition("_")
        try:
            expected = instructions()[suite][int(index)]
        except (KeyError, ValueError, IndexError):
            problems.append(f"{key}: not a task of the four suites")
            continue
        if not _normal(text).endswith(_normal(expected)):
            problems.append(f"{key}: LIBERO says {text!r}, the table says {expected!r}")
    return problems


def _normal(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split())
