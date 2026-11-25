"""LIBERO suite names, episode step limits and the instruction table the arms rely on."""

import json
from functools import cache
from importlib import resources

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS_PER_SUITE = 10

# Episode step limits of LeRobot 0.7.0 (TASK_SUITE_MAX_STEPS in sim/integrations/libero.py).
STEP_CAPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def read_data(name: str) -> dict:
    path = resources.files("vlashift").joinpath("data", name)
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def instructions() -> dict[str, tuple[str, ...]]:
    table = read_data("libero_instructions.json")
    return {suite: tuple(table[suite]) for suite in SUITES}


@cache
def _index() -> dict[str, tuple[str, int]]:
    return {
        text: (suite, position)
        for suite, texts in instructions().items()
        for position, text in enumerate(texts)
    }


def locate(text: str) -> tuple[str, int]:
    """Suite and task index of an exact LIBERO instruction."""
    try:
        return _index()[text]
    except KeyError:
        raise KeyError(f"not an instruction of the four LIBERO suites: {text!r}") from None
