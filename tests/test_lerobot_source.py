"""Tests against the source of an installed LeRobot: no import of it, no weights, no simulator.

Run with LEROBOT_SRC=<site-packages>/lerobot pytest -m needs_lerobot
"""

import ast
import importlib.util
import re

import pytest

from vlashift.patch import ANCHORS, build, eval_script, lerobot_dir
from vlashift.runtime import IMAGE_PREFIX
from vlashift.stats import wilson
from vlashift.suites import STEP_CAPS

pytestmark = pytest.mark.needs_lerobot


@pytest.fixture(scope="module")
def root():
    try:
        return lerobot_dir()
    except FileNotFoundError:
        pytest.skip("no LeRobot source: set LEROBOT_SRC")


def test_every_anchor_occurs_once_in_the_installed_script(root):
    source = eval_script().read_text()
    assert [source.count(old) for _, old, _ in ANCHORS] == [1] * len(ANCHORS)


def test_the_patched_script_compiles(root):
    compile(build(eval_script().read_text()), "patched.py", "exec")


def lerobot_constants(root):
    values = {}
    for node in ast.parse((root / "utils" / "constants.py").read_text()).body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            try:
                code = compile(ast.Expression(node.value), "constants", "eval")
                values[target.id] = eval(code, {"__builtins__": {}}, values)
            except Exception:  # noqa: S110 - names imported from elsewhere, calls
                pass
    return values


def test_image_prefix_is_the_one_lerobot_uses(root):
    assert lerobot_constants(root)["OBS_IMAGES"] + "." == IMAGE_PREFIX


def test_step_caps_are_the_ones_in_the_libero_integration(root):
    text = (root / "sim" / "integrations" / "libero.py").read_text()
    found = {m.group(1): int(m.group(2)) for m in re.finditer(r'"(libero_\w+)":\s*(\d+),', text)}
    assert {k: found[k] for k in STEP_CAPS} == STEP_CAPS


def test_output_keys_the_parser_reads_are_written_by_the_script(root):
    text = eval_script().read_text()
    for key in (
        '"per_task"',
        '"task_group"',
        '"task_id"',
        '"metrics"',
        '"successes"',
        '"eval_ep_s"',
    ):
        assert key in text


def test_wilson_agrees_with_the_lerobot_implementation(root):
    spec = importlib.util.spec_from_file_location(
        "lerobot_eval_stats", root / "utils" / "eval_stats.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for n in (1, 5, 10, 50, 100, 400):
        for k in range(0, n + 1, max(1, n // 10)):
            assert wilson(k, n) == pytest.approx(module.wilson_interval(k, n), abs=1e-12)
