import json

import pytest
from helpers import flags, uniform, write_run

from vlashift import cli
from vlashift.config import STAGES, parse_arm, shell_lines

SAMPLE = (
    "import logging\n\nlogger = logging.getLogger(__name__)\n"
    'observation["task"] = list(env.call("task_description"))\n'
    "observation = env_preprocessor(observation)\n"
    "max_episodes_rendered = 0 if cfg.eval.recording else 10\n"
    "def main():\n    init_logging()\n    register_third_party_plugins()\n    eval_main()\n"
)


def run(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    return code, capsys.readouterr().out


def test_plan_counts_episodes_per_stage():
    rows = {r["stage"]: r for r in cli.plan_rows(10.0)}
    assert [rows[s]["episodes"] for s in STAGES] == [400, 1600, 1600, 2000]
    assert [rows[s]["arms"] for s in STAGES] == [1, 4, 4, 5]


def test_plan_hours_scale_with_the_step_limit_of_each_suite():
    one = cli.plan_rows(10.0, suites=["libero_spatial"])[0]["hours"]
    assert one == pytest.approx(10.0 * 100 / 3600)
    long = cli.plan_rows(10.0, suites=["libero_10"])[0]["hours"]
    assert long == pytest.approx(one * 520 / 280)
    every = cli.plan_rows(10.0)[0]["hours"]
    assert every == pytest.approx(10.0 * 100 * (1 + 1 + 300 / 280 + 520 / 280) / 3600)


def test_plan_hours_follow_seconds_and_episodes_per_task():
    base = cli.plan_rows(10.0)[1]["hours"]
    assert cli.plan_rows(20.0)[1]["hours"] == pytest.approx(2 * base)
    assert cli.plan_rows(10.0, episodes_per_task=5)[1]["hours"] == pytest.approx(base / 2)


def test_plan_command_prints_a_cumulative_table(capsys):
    code, out = run(capsys, "plan", "--sec-per-episode", "10")
    assert code == 0
    lines = out.strip().splitlines()
    assert lines[0].startswith("| stage | arms | episodes | hours | cumulative hours |")
    assert lines[2].startswith("| baseline | 1 | 400 | 1.4 | 1.4 |")
    assert len(lines) == 2 + len(STAGES)


def test_plan_command_can_restrict_the_suites(capsys):
    code, out = run(capsys, "plan", "--sec-per-episode", "10", "--suites", "libero_goal")
    assert code == 0
    assert "| baseline | 1 | 100 |" in out


def test_arm_env_prints_clearing_and_export_lines(capsys):
    code, out = run(capsys, "arm-env", "joint_0.2")
    assert code == 0
    assert out.splitlines() == shell_lines(parse_arm("joint_0.2"))
    assert "export VLASHIFT_JOINT_STD=0.2" in out


def test_arm_env_rejects_an_unknown_arm(capsys):
    with pytest.raises(ValueError, match="unknown arm"):
        cli.main(["arm-env", "fog_0.5"])


def test_arms_lists_stage_members_in_order(capsys):
    code, out = run(capsys, "arms", "baseline", "instruction")
    assert out.split() == ["keep", "blank", "other", "shuffled", "paraphrase"]
    assert run(capsys, "arms", "shifts")[1].split() == list(STAGES["shifts"])


def test_arms_rejects_an_unknown_stage(capsys):
    with pytest.raises(SystemExit):
        cli.main(["arms", "everything"])


def test_build_eval_writes_the_patched_copy(tmp_path, capsys):
    source = tmp_path / "lerobot_eval.py"
    source.write_text(SAMPLE)
    out = tmp_path / "patched" / "eval.py"
    code, text = run(capsys, "build-eval", "--source", source, "--out", out)
    assert code == 0
    assert "_vlashift.install()" in out.read_text()
    assert str(out) in text


def test_check_applied_exit_code_follows_the_problems(tmp_path, capsys):
    record = {
        "config": {"instruction": "blank", "brightness": None, "noise_std": None,
                   "occlusion": None, "joint_std": None},
        "counts": {"instructions_changed": 3, "images_perturbed": 0, "init_states_perturbed": 0},
        "first_states": [],
    }  # fmt: skip
    (tmp_path / "vlashift_applied.json").write_text(json.dumps(record))
    assert run(capsys, "check-applied", tmp_path, "blank")[0] == 0
    code, out = run(capsys, "check-applied", tmp_path, "keep")
    assert code == 1
    assert "changed 3 times" in out


def test_check_descriptions_reports_mismatches(tmp_path, capsys):
    (tmp_path / "task_descriptions.json").write_text(
        json.dumps({"libero_goal_7": "turn on the stove"})
    )
    assert run(capsys, "check-descriptions", tmp_path)[0] == 0
    (tmp_path / "task_descriptions.json").write_text(json.dumps({"libero_goal_7": "close it"}))
    code, out = run(capsys, "check-descriptions", tmp_path)
    assert code == 1
    assert "libero_goal_7" in out


def test_show_prints_one_line_per_run(tmp_path, capsys):
    first = write_run(tmp_path, "keep", "libero_goal", {0: flags(9), 1: flags(7)}, eval_ep_s=9.25)
    second = write_run(tmp_path, "blank", "libero_object", uniform(1, tasks=2))
    code, out = run(capsys, "show", first, second)
    assert code == 0
    assert out.splitlines() == [
        "keep/libero_goal: 16/20 episodes succeeded, 9.2 s/episode",
        "blank/libero_object: 2/20 episodes succeeded, 7.5 s/episode",
    ]


def test_meta_records_the_protocol(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("VLASHIFT_NOISE_STD", "0.1")
    monkeypatch.setenv("HOME_SECRET", "no")
    code, _ = run(
        capsys,
        "meta",
        tmp_path,
        "--arm",
        "noise_0.1",
        "--suite",
        "libero_goal",
        "--revision",
        "abc",
        "--reset",
        "hard",
        "--extra-args",
        "",
        "--episodes",
        "10",
        "--seed",
        "1000",
        "--image-size",
        "256",
    )
    assert code == 0
    info = json.loads((tmp_path / "meta.json").read_text())
    assert (info["arm"], info["suite"], info["episodes_per_task"]) == (
        "noise_0.1",
        "libero_goal",
        10,
    )
    assert (info["seed"], info["image_size"], info["reset"], info["task_ids"]) == (
        1000,
        256,
        "hard",
        "all",
    )
    assert info["env"] == {"VLASHIFT_NOISE_STD": "0.1"}
    assert info["versions"]["numpy"]
    assert "written" in info


def test_versions_lists_none_for_missing_packages():
    found = cli.versions()
    assert found["lerobot"] is None or isinstance(found["lerobot"], str)
    assert set(found) >= {"lerobot", "torch", "numpy", "hf-libero"}
