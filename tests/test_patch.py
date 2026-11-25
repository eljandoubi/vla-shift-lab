import hashlib

import pytest

from vlashift import patch
from vlashift.patch import ANCHORS, AnchorError, build, build_file, lerobot_dir

SAMPLE = '''"""Evaluate a policy."""

import logging

logger = logging.getLogger(__name__)


def rollout(env, preprocessor, env_preprocessor):
    task_desc = list(env.call("task_description"))[0]
    while True:
        try:
            observation["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            observation["task"] = [""]
        observation = env_preprocessor(observation)
        observation = preprocessor(observation)


def eval_main(cfg):
    max_episodes_rendered = 0 if cfg.eval.recording else 10
    return max_episodes_rendered


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
'''


def test_build_replaces_each_anchor_once():
    patched = build(SAMPLE)
    assert "import vlashift.runtime as _vlashift" in patched
    assert (
        'observation["task"] = _vlashift.instructions(list(env.call("task_description")))'
        in patched
    )
    assert "observation = _vlashift.images(env_preprocessor(observation))" in patched
    assert "else _vlashift.video_episodes()" in patched
    assert "    _vlashift.install()\n    eval_main()" in patched


def test_build_leaves_the_other_task_description_line_alone():
    patched = build(SAMPLE)
    assert 'task_desc = list(env.call("task_description"))[0]' in patched
    assert patched.count("_vlashift.instructions(") == 1


def test_build_keeps_the_rest_of_the_script_and_compiles():
    patched = build(SAMPLE)
    compile(patched, "patched.py", "exec")
    added = patched.count("\n") - SAMPLE.count("\n")
    assert added == 4  # two header lines, the import, the install call
    assert 'if __name__ == "__main__":' in patched


def test_build_starts_with_the_digest_of_the_original():
    patched = build(SAMPLE)
    assert patched.startswith("# Patched copy of lerobot/scripts/lerobot_eval.py")
    assert hashlib.sha256(SAMPLE.encode()).hexdigest() in patched.splitlines()[0]


@pytest.mark.parametrize("anchor", [name for name, _, _ in ANCHORS])
def test_a_missing_anchor_stops_the_build(anchor):
    _, old, _ = next(a for a in ANCHORS if a[0] == anchor)
    broken = SAMPLE.replace(old, "# removed\n" if old.endswith("\n") else "pass")
    with pytest.raises(AnchorError, match=f"anchor '{anchor}' occurs 0 times"):
        build(broken)


@pytest.mark.parametrize("anchor", [name for name, _, _ in ANCHORS])
def test_a_repeated_anchor_stops_the_build(anchor):
    _, old, _ = next(a for a in ANCHORS if a[0] == anchor)
    with pytest.raises(AnchorError, match=f"anchor '{anchor}' occurs 2 times"):
        build(SAMPLE + "\n" + old + "\n")


def test_build_file_writes_a_compiled_copy(tmp_path):
    source = tmp_path / "lerobot_eval.py"
    source.write_text(SAMPLE)
    out = tmp_path / "nested" / "eval_patched.py"
    text = build_file(out, source)
    assert out.read_text() == text


def test_build_file_refuses_a_script_that_does_not_compile(tmp_path, monkeypatch):
    source = tmp_path / "lerobot_eval.py"
    source.write_text(SAMPLE)
    monkeypatch.setattr(patch, "build", lambda text: "def broken(:\n")
    with pytest.raises(SyntaxError):
        patch.build_file(tmp_path / "out.py", source)
    assert not (tmp_path / "out.py").exists()


def test_lerobot_dir_follows_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("LEROBOT_SRC", str(tmp_path))
    assert lerobot_dir() == tmp_path


def test_lerobot_dir_reports_a_missing_install(monkeypatch):
    monkeypatch.delenv("LEROBOT_SRC", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(FileNotFoundError, match="LEROBOT_SRC"):
        lerobot_dir()
