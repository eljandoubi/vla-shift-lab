import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COLAB = ROOT / "colab"
SCRIPTS = sorted(COLAB.glob("*.sh"))
REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"

STUB = """
import json
import os
import sys
from pathlib import Path

import numpy as np

from vlashift import runtime

args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
with Path(os.environ["STUB_LOG"]).open("a") as handle:
    handle.write(args["env.task"] + " " + os.environ.get("VLASHIFT_INSTRUCTION", "") + "\\n")
if os.environ.get("FAKE_FAIL"):
    sys.exit(3)
out = Path(args["output_dir"])
out.mkdir(parents=True, exist_ok=True)
if not os.environ.get("FAKE_NO_HOOK"):
    runtime.install(at_exit=True)
    runtime.instructions(["turn on the stove"])
    runtime.images({"observation.images.image": np.full((1, 3, 8, 8), 0.5, dtype=np.float32)})
task = {"task_group": args["env.task"], "task_id": 0, "metrics": {"successes": [True, False]}}
info = {"per_task": [task], "overall": {"eval_ep_s": 2.5}}
(out / "eval_info.json").write_text(json.dumps(info))
"""


def run(tmp_path, script, *args, **env):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("VLASHIFT_")}
    environment.update(PY=sys.executable, OUT=str(tmp_path / "out"), LR=str(tmp_path / "lr"))
    environment.update({k: str(v) for k, v in env.items()})
    return subprocess.run(
        ["bash", str(COLAB / script), *args],
        capture_output=True,
        text=True,
        env=environment,
        cwd=tmp_path,
        check=False,
    )


@pytest.fixture
def stub(tmp_path):
    path = tmp_path / "stub_eval.py"
    path.write_text(STUB)
    return {"PATCHED": path, "STUB_LOG": tmp_path / "stub.log"}


def dry_lines(done):
    return [line for line in done.stdout.splitlines() if line.startswith("DRY ")]


def test_common_replaces_the_notebook_plotting_backend():
    """Colab exports a notebook backend that matplotlib in the virtualenv rejects."""
    done = subprocess.run(
        ["bash", "-c", f'. "{COLAB / "common.sh"}"; echo $MPLBACKEND'],
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "module://matplotlib_inline.backend_inline"},
        check=False,
    )
    assert done.stdout.splitlines()[-1] == "Agg"


def test_setup_pins_mujoco_below_the_release_that_breaks_libero_spatial():
    assert '"mujoco==3.3.7"' in (COLAB / "setup.sh").read_text()


def test_egl_script_writes_the_vendor_file_for_the_nvidia_library(tmp_path):
    lib = tmp_path / "libEGL_nvidia.so.0"
    lib.touch()
    vendor = tmp_path / "vendor"
    done = run(tmp_path, "egl.sh", NVIDIA_EGL_LIB=lib, EGL_VENDOR_DIR=vendor)
    assert done.returncode == 0
    record = json.loads((vendor / "10_nvidia.json").read_text())
    assert record["ICD"]["library_path"] == str(lib)


def test_egl_script_does_nothing_without_the_library(tmp_path):
    vendor = tmp_path / "vendor"
    done = run(tmp_path, "egl.sh", NVIDIA_EGL_LIB=tmp_path / "missing.so", EGL_VENDOR_DIR=vendor)
    assert done.returncode == 0 and "stays on the CPU" in done.stdout
    assert not vendor.exists()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_scripts_are_valid_bash(script):
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_dry_run_prints_one_command_per_arm_and_suite(tmp_path):
    done = run(
        tmp_path,
        "run_all.sh",
        DRY_RUN=1,
        STAGES="baseline instruction",
        SUITES="libero_spatial libero_goal",
    )
    assert done.returncode == 0, done.stderr
    lines = dry_lines(done)
    assert len(lines) == 10
    blank = next(line for line in lines if line.startswith("DRY blank/libero_goal"))
    out = tmp_path / "out"
    for part in (
        "VLASHIFT_INSTRUCTION=blank",
        "--policy.path=lerobot/smolvla_libero",
        f"--policy.pretrained_revision={REVISION}",
        "--env.type=libero",
        "--env.task=libero_goal",
        "--env.observation_height=256",
        "--env.observation_width=256",
        "--env.init_states=true",
        "--eval.n_episodes=10",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        "--policy.n_action_steps=10",
        "--seed=1000",
        f"--output_dir={out}/blank/libero_goal.partial",
    ):
        assert part in blank, part
    assert "--env.hard_reset" not in blank
    assert "--env.task_ids" not in blank
    assert "VLASHIFT_APPLIED_PATH" not in blank


def test_dry_run_options_reach_the_command(tmp_path):
    done = run(
        tmp_path,
        "run_all.sh",
        DRY_RUN=1,
        ARMS="keep",
        SUITES="libero_object",
        RESET="soft",
        TASK_IDS="[0]",
        EPISODES=3,
        IMAGE_SIZE=360,
        SEED=7,
        N_ACTION_STEPS=25,
        EXTRA_ARGS="--policy.dtype=torch.float32 --foo=1",
    )
    (line,) = dry_lines(done)
    for part in (
        "--env.hard_reset=false",
        "--env.task_ids=[0]",
        "--eval.n_episodes=3",
        "--env.observation_height=360",
        "--seed=7",
        "--policy.n_action_steps=25",
        "--policy.dtype=torch.float32",
        "--foo=1",
    ):
        assert part in line, part


def test_dry_run_sets_exactly_the_variables_of_each_arm(tmp_path):
    done = run(
        tmp_path,
        "run_all.sh",
        DRY_RUN=1,
        ARMS="joint_0.2 bright_0.5 paraphrase",
        SUITES="libero_goal",
    )
    joint, bright, paraphrase = dry_lines(done)
    assert "VLASHIFT_JOINT_STD=0.2" in joint
    assert "VLASHIFT_INSTRUCTION=keep" in joint
    assert "VLASHIFT_BRIGHTNESS" not in joint
    assert "VLASHIFT_BRIGHTNESS=0.5" in bright
    assert "VLASHIFT_JOINT_STD" not in bright
    assert "VLASHIFT_INSTRUCTION=paraphrase" in paraphrase
    assert "VLASHIFT_BRIGHTNESS" not in paraphrase


def test_a_finished_run_is_skipped(tmp_path):
    finished = tmp_path / "out" / "keep" / "libero_spatial"
    finished.mkdir(parents=True)
    (finished / "eval_info.json").write_text("{}")
    done = run(tmp_path, "run_all.sh", DRY_RUN=1, ARMS="keep", SUITES="libero_spatial libero_goal")
    assert "skip keep/libero_spatial (finished)" in done.stdout
    assert [line.split(":")[0] for line in dry_lines(done)] == ["DRY keep/libero_goal"]


def test_a_directory_without_eval_info_is_not_finished(tmp_path):
    (tmp_path / "out" / "keep" / "libero_spatial").mkdir(parents=True)
    done = run(tmp_path, "run_all.sh", DRY_RUN=1, ARMS="keep", SUITES="libero_spatial")
    assert len(dry_lines(done)) == 1


def test_an_unknown_arm_fails_the_whole_call(tmp_path):
    done = run(tmp_path, "run_all.sh", DRY_RUN=1, ARMS="keep fog_0.5", SUITES="libero_goal")
    assert done.returncode == 1
    assert "FAILED: fog_0.5/libero_goal" in done.stdout
    assert len(dry_lines(done)) == 1


def test_a_run_ends_as_a_finished_directory_with_its_record(tmp_path, stub):
    done = run(tmp_path, "run_all.sh", ARMS="keep blank noise_0.1", SUITES="libero_goal", **stub)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "all runs finished" in done.stdout
    for arm in ("keep", "blank", "noise_0.1"):
        directory = tmp_path / "out" / arm / "libero_goal"
        assert {p.name for p in directory.iterdir()} >= {
            "eval_info.json",
            "vlashift_applied.json",
            "meta.json",
            "run.log",
            "check.log",
        }
        assert not (tmp_path / "out" / arm / "libero_goal.partial").exists()
    meta = json.loads((tmp_path / "out" / "blank" / "libero_goal" / "meta.json").read_text())
    assert (meta["arm"], meta["suite"], meta["revision"], meta["image_size"]) == (
        "blank",
        "libero_goal",
        REVISION,
        256,
    )
    assert meta["env"]["VLASHIFT_INSTRUCTION"] == "blank"


def test_starting_again_skips_what_is_finished(tmp_path, stub):
    run(tmp_path, "run_all.sh", ARMS="keep blank", SUITES="libero_goal", **stub)
    calls = (tmp_path / "stub.log").read_text().splitlines()
    assert calls == ["libero_goal keep", "libero_goal blank"]
    done = run(tmp_path, "run_all.sh", ARMS="keep blank noise_0.1", SUITES="libero_goal", **stub)
    assert done.stdout.count("skip ") == 2
    assert (tmp_path / "stub.log").read_text().splitlines()[2:] == ["libero_goal keep"]


def test_a_crashed_run_is_kept_as_failed_and_the_others_continue(tmp_path, stub):
    done = run(tmp_path, "run_all.sh", ARMS="keep blank", SUITES="libero_goal", FAKE_FAIL=1, **stub)
    assert done.returncode == 1
    assert "FAILED: keep/libero_goal blank/libero_goal" in done.stdout
    failed = tmp_path / "out" / "keep" / "libero_goal.failed"
    assert (failed / "run.log").exists()
    assert not (tmp_path / "out" / "keep" / "libero_goal").exists()


def test_a_run_whose_hooks_never_ran_is_rejected(tmp_path, stub):
    done = run(tmp_path, "run_one.sh", "blank", "libero_goal", FAKE_NO_HOOK=1, **stub)
    assert done.returncode == 1
    assert "the hooks did not do what the arm needs" in done.stdout
    assert (tmp_path / "out" / "blank" / "libero_goal.failed" / "eval_info.json").exists()
    assert not (tmp_path / "out" / "blank" / "libero_goal").exists()


def test_a_failed_run_is_run_again_from_scratch(tmp_path, stub):
    run(tmp_path, "run_one.sh", "keep", "libero_goal", FAKE_FAIL=1, **stub)
    assert (tmp_path / "out" / "keep" / "libero_goal.failed").exists()
    done = run(tmp_path, "run_one.sh", "keep", "libero_goal", **stub)
    assert done.returncode == 0
    assert not (tmp_path / "out" / "keep" / "libero_goal.failed").exists()
    assert (tmp_path / "out" / "keep" / "libero_goal" / "eval_info.json").exists()


def test_smoke_dry_run_covers_every_hook_with_one_episode(tmp_path):
    done = run(tmp_path, "smoke.sh", DRY_RUN=1)
    assert done.returncode == 0, done.stdout + done.stderr
    lines = dry_lines(done)
    assert [line.split(":")[0].split()[1] for line in lines] == [
        "keep/libero_spatial",
        "blank/libero_spatial",
        "other/libero_spatial",
        "shuffled/libero_spatial",
        "paraphrase/libero_spatial",
        "bright_0.5/libero_spatial",
        "noise_0.1/libero_spatial",
        "occl_0.4/libero_spatial",
        "joint_0.2/libero_spatial",
    ]
    assert all("--eval.n_episodes=1 " in line and "--env.task_ids=[0]" in line for line in lines)
    assert all("VLASHIFT_DUMP_FRAMES=2" in line for line in lines)
    assert "smoke" in lines[0]
    assert done.stdout.strip().endswith("SMOKE OK")


def test_collect_packs_small_files_only(tmp_path):
    out = tmp_path / "out"
    for name in (
        "keep/libero_goal/eval_info.json",
        "keep/libero_goal/meta.json",
        "keep/libero_goal/frames/frame0_image.png",
        "keep/libero_goal/videos/libero_goal_0/eval_episode_0.mp4",
        "keep/libero_goal/checkpoints/model.safetensors",
        "blank/libero_goal.failed/run.log",
        "notes.txt",
    ):
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    archive = tmp_path / "result.tar.gz"
    done = run(tmp_path, "collect.sh", str(archive))
    assert done.returncode == 0, done.stderr
    with tarfile.open(archive) as tar:
        names = sorted(tar.getnames())
    assert names == [
        "./blank/libero_goal.failed/run.log",
        "./keep/libero_goal/eval_info.json",
        "./keep/libero_goal/frames/frame0_image.png",
        "./keep/libero_goal/meta.json",
    ]
    assert "4 files" in done.stdout


def test_collect_without_results_says_so(tmp_path):
    (tmp_path / "out").mkdir()
    done = run(tmp_path, "collect.sh", str(tmp_path / "x.tar.gz"))
    assert done.returncode == 1
    assert "no result files" in done.stdout
