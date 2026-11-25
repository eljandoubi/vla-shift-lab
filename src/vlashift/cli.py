"""Command line entry points used by the Colab scripts."""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from vlashift import patch
from vlashift.checks import applied_problems, description_problems, read_applied
from vlashift.config import KNOWN_ENV, STAGES, parse_arm, shell_lines
from vlashift.results import read_run
from vlashift.suites import STEP_CAPS, SUITES, TASKS_PER_SUITE


def plan_rows(seconds_per_episode: float, episodes_per_task: int = 10, suites=SUITES) -> list[dict]:
    """Episodes and hours per stage; a suite costs in proportion to its step limit."""
    reference = STEP_CAPS["libero_spatial"]
    rows = []
    for stage, arms in STAGES.items():
        per_suite = len(arms) * TASKS_PER_SUITE * episodes_per_task
        seconds = sum(per_suite * seconds_per_episode * STEP_CAPS[s] / reference for s in suites)
        rows.append(
            {
                "stage": stage,
                "arms": len(arms),
                "episodes": per_suite * len(suites),
                "hours": seconds / 3600,
            }
        )
    return rows


def plan_table(rows: list[dict]) -> str:
    lines = ["| stage | arms | episodes | hours | cumulative hours |", "|---|---|---|---|---|"]
    total = 0.0
    for r in rows:
        total += r["hours"]
        lines.append(
            f"| {r['stage']} | {r['arms']} | {r['episodes']} | {r['hours']:.1f} | {total:.1f} |"
        )
    return "\n".join(lines)


def versions() -> dict[str, str | None]:
    found = {}
    for name in ("vla-shift-lab", "lerobot", "torch", "numpy", "hf-libero", "robosuite", "mujoco"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def gpu_name() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def cmd_build_eval(args) -> int:
    text = patch.build_file(args.out, args.source)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    return 0


def cmd_arm_env(args) -> int:
    print("\n".join(shell_lines(parse_arm(args.arm))))
    return 0


def cmd_arms(args) -> int:
    names = [name for stage in args.stage for name in STAGES[stage]]
    print("\n".join(names))
    return 0


def cmd_plan(args) -> int:
    suites = args.suites or SUITES
    print(plan_table(plan_rows(args.sec_per_episode, args.episodes_per_task, suites)))
    return 0


def cmd_check_applied(args) -> int:
    baseline = read_applied(args.baseline) if args.baseline else None
    problems = applied_problems(parse_arm(args.arm), read_applied(args.directory), baseline)
    for problem in problems:
        print(f"{args.directory}: {problem}")
    return 1 if problems else 0


def cmd_check_descriptions(args) -> int:
    problems = description_problems(args.directory)
    for problem in problems:
        print(f"{args.directory}: {problem}")
    return 1 if problems else 0


def cmd_show(args) -> int:
    for directory in args.directories:
        run = read_run(directory)
        seconds = (
            f"{run.seconds_per_episode:.1f} s/episode" if run.seconds_per_episode else "no timing"
        )
        record = Path(directory) / "vlashift_applied.json"
        first = json.loads(record.read_text())["first_states"] if record.is_file() else []
        start = f", start eef {first[0][:3]}" if first else ""
        print(
            f"{run.arm}/{run.suite}: {run.successes}/{run.episodes} episodes succeeded, "
            f"{seconds}{start}"
        )
    return 0


def cmd_meta(args) -> int:
    info = {
        "arm": args.arm,
        "suite": args.suite,
        "episodes_per_task": args.episodes,
        "seed": args.seed,
        "revision": args.revision,
        "image_size": args.image_size,
        "reset": args.reset,
        "task_ids": args.task_ids,
        "extra_args": args.extra_args,
        "python": sys.version.split()[0],
        "versions": versions(),
        "gpu": gpu_name(),
        "env": {k: v for k, v in sorted(os.environ.items()) if k in KNOWN_ENV},
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (args.directory / "meta.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlashift", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, function, help_text):
        sub = commands.add_parser(name, help=help_text)
        sub.set_defaults(run=function)
        return sub

    sub = add("build-eval", cmd_build_eval, "write the patched copy of lerobot_eval.py")
    sub.add_argument("--out", type=Path, required=True)
    sub.add_argument("--source", type=Path, help="original script, default: the installed one")

    sub = add("arm-env", cmd_arm_env, "shell lines that set the variables of one arm")
    sub.add_argument("arm")

    sub = add("arms", cmd_arms, "arm names of one or more stages")
    sub.add_argument("stage", nargs="+", choices=list(STAGES))

    sub = add("plan", cmd_plan, "episodes and hours per stage")
    sub.add_argument("--sec-per-episode", type=float, required=True, help="libero_spatial episode")
    sub.add_argument("--episodes-per-task", type=int, default=10)
    sub.add_argument("--suites", nargs="+", choices=SUITES)

    sub = add("check-applied", cmd_check_applied, "did the hooks do what the arm needs")
    sub.add_argument("directory", type=Path)
    sub.add_argument("arm")
    sub.add_argument("--baseline", type=Path, help="keep run, to compare the start pose")

    sub = add("check-descriptions", cmd_check_descriptions, "LIBERO task text against the table")
    sub.add_argument("directory", type=Path)

    sub = add("show", cmd_show, "success count and speed of finished run directories")
    sub.add_argument("directories", nargs="+", type=Path)

    sub = add("meta", cmd_meta, "write meta.json next to a run")
    sub.add_argument("directory", type=Path)
    for flag in ("arm", "suite", "revision", "reset", "extra-args"):
        sub.add_argument(f"--{flag}", required=True)
    sub.add_argument("--episodes", type=int, required=True)
    sub.add_argument("--seed", type=int, required=True)
    sub.add_argument("--image-size", type=int, required=True)
    sub.add_argument("--task-ids", default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
