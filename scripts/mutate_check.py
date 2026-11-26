"""Hand-made faults, applied one at a time to a copy of the sources, run against the fast tests.

A mutant is killed when the tests fail with the fault in place. The repository itself is never
touched: sources, tests and pyproject.toml are copied to a temporary folder first.

    python scripts/mutate_check.py --out results/mutants.txt
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

STATS = "src/vlashift/stats.py"
INSTRUCTIONS = "src/vlashift/instructions.py"
RESULTS = "src/vlashift/results.py"
PERTURB = "src/vlashift/perturb.py"

# name, file, text that must occur exactly once, replacement
MUTANTS = [
    (
        "wilson_centre_unshifted",
        STATS,
        "centre = (p + z * z / (2 * trials)) / denominator",
        "centre = p / denominator",
    ),
    (
        "wilson_not_clamped",
        STATS,
        "return (max(0.0, centre - half), min(1.0, centre + half))",
        "return (centre - half, centre + half)",
    ),
    (
        "wilson_wrong_z",
        STATS,
        "z = NormalDist().inv_cdf(0.5 + confidence / 2)",
        "z = NormalDist().inv_cdf(confidence)",
    ),
    (
        "mcnemar_tail_off_by_one",
        STATS,
        "range(min(only_a, only_b) + 1)",
        "range(min(only_a, only_b))",
    ),
    (
        "mcnemar_one_sided",
        STATS,
        "return min(1.0, 2 * tail / 2**discordant)",
        "return min(1.0, tail / 2**discordant)",
    ),
    (
        "pairs_roles_swapped",
        STATS,
        "only_a = sum(1 for k in shared if a[k] and not b[k])",
        "only_a = sum(1 for k in shared if b[k] and not a[k])",
    ),
    (
        "tasks_grouped_by_first_key_part",
        STATS,
        "groups[key[:-1]].append(key)",
        "groups[key[0]].append(key)",
    ),
    (
        "bootstrap_upper_bound_is_lower",
        STATS,
        "float(np.quantile(means, 1 - alpha))",
        "float(np.quantile(means, alpha))",
    ),
    (
        "contrast_adds_means",
        STATS,
        "return float(a.mean() - b.mean()), low, high",
        "return float(a.mean() + b.mean()), low, high",
    ),
    (
        "other_goes_backwards",
        INSTRUCTIONS,
        "return texts[(position + 1) % len(texts)]",
        "return texts[(position - 1) % len(texts)]",
    ),
    (
        "other_wraps_too_early",
        INSTRUCTIONS,
        "return texts[(position + 1) % len(texts)]",
        "return texts[(position + 1) % (len(texts) - 1)]",
    ),
    ("shuffle_may_keep_order", INSTRUCTIONS, "if candidate != text:", "if candidate == text:"),
    ("shuffle_ignores_position", INSTRUCTIONS, 'f"{salt}|{text}|{i}"', 'f"{salt}|{text}"'),
    ("blank_keeps_text", INSTRUCTIONS, '    return ""', "    return text"),
    ("paraphrase_returns_original", INSTRUCTIONS, "return paraphrases()[text]", "return text"),
    (
        "every_episode_succeeds",
        RESULTS,
        "outcomes[(task_id, episode)] = bool(flag)",
        "outcomes[(task_id, episode)] = True",
    ),
    (
        "episodes_count_from_one",
        RESULTS,
        "for episode, flag in enumerate(flags):",
        "for episode, flag in enumerate(flags, 1):",
    ),
    ("suite_mismatch_accepted", RESULTS, "if group != suite:", "if group == suite:"),
    (
        "brightness_not_clipped",
        PERTURB,
        "return np.clip(images * scale, 0.0, 1.0)",
        "return images * scale",
    ),
    (
        "occlusion_in_corner",
        PERTURB,
        "top, left = (height - side) // 2, (width - side) // 2",
        "top, left = 0, 0",
    ),
    ("joint_noise_hits_time", PERTURB, "ARM_JOINTS = slice(1, 8)", "ARM_JOINTS = slice(0, 7)"),
    (
        "joint_noise_ignores_std",
        PERTURB,
        "out[row, ARM_JOINTS] += std * draw",
        "out[row, ARM_JOINTS] += draw",
    ),
    (
        "changed_count_inverted",
        "src/vlashift/runtime.py",
        "a != b for a, b in zip(before, after, strict=True)",
        "a == b for a, b in zip(before, after, strict=True)",
    ),
    (
        "delta_sign_flipped",
        "src/vlashift/report.py",
        "per_task_differences(base, other)",
        "per_task_differences(other, base)",
    ),
    (
        "rate_not_in_percent",
        "src/vlashift/report.py",
        "rate = 100 * successes / trials",
        "rate = successes / trials",
    ),
    (
        "no_start_pose_check",
        "src/vlashift/checks.py",
        "if gap < MIN_EEF_SHIFT:",
        "if gap > MIN_EEF_SHIFT:",
    ),
    ("zero_level_accepted", "src/vlashift/config.py", "if not level > 0:", "if not level >= 0:"),
    (
        "images_hook_before_preprocessor",
        "src/vlashift/patch.py",
        '"observation = _vlashift.images(env_preprocessor(observation))",',
        '"observation = env_preprocessor(_vlashift.images(observation))",',
    ),
    (
        "h1_cost_threshold_loosened",
        "src/vlashift/report.py",
        "if worst < 15 and goal - worst >= 15:",
        "if worst < 16 and goal - worst >= 15:",
    ),
    (
        "h1_goal_gap_shrunk",
        "src/vlashift/report.py",
        "goal - worst >= 15",
        "goal - worst >= 14",
    ),
    (
        "h1_falsifier_threshold_raised",
        "src/vlashift/report.py",
        "if worst >= 30 or goal <= worst:",
        "if worst >= 31 or goal <= worst:",
    ),
    (
        "h1_goal_tie_passes",
        "src/vlashift/report.py",
        "if worst >= 30 or goal <= worst:",
        "if worst >= 30 or goal < worst:",
    ),
    (
        "h2_holds_threshold_loosened",
        "src/vlashift/report.py",
        'return "holds" if worst < 10 else "unclear"',
        'return "holds" if worst < 11 else "unclear"',
    ),
    (
        "h2_falsifier_threshold_raised",
        "src/vlashift/report.py",
        "if worst >= 20:",
        "if worst > 20:",
    ),
    (
        "h3_touching_intervals_hold",
        "src/vlashift/report.py",
        "if min(noise[1], occl[1]) > bright[2]:",
        "if min(noise[1], occl[1]) >= bright[2]:",
    ),
    (
        "h3_brightness_tie_passes",
        "src/vlashift/report.py",
        "if bright[0] >= min(noise[0], occl[0]):",
        "if bright[0] > min(noise[0], occl[0]):",
    ),
    (
        "h3_checks_only_noise",
        "src/vlashift/report.py",
        "if min(noise[1], occl[1]) > bright[2]:",
        "if noise[1] > bright[2]:",
    ),
    (
        "h3_arms_on_different_episodes",
        "src/vlashift/report.py",
        'same_episodes = len(image) == 3 and len({r["pairs"] for r in image.values()}) == 1',
        "same_episodes = len(image) == 3",
    ),
    (
        "hypothesis_delta_sign_flipped",
        "src/vlashift/report.py",
        "return 100 * (counts.only_a - counts.only_b) / counts.pairs",
        "return 100 * (counts.only_b - counts.only_a) / counts.pairs",
    ),
]

TEST_COMMAND = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"]
TEST_COMMAND += ["--ignore=tests/test_colab.py"]


def apply(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise ValueError(f"{path}: expected one match, found {text.count(old)}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def run_tests(root: Path) -> tuple[bool, str]:
    env = {"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"}
    done = subprocess.run(
        TEST_COMMAND, cwd=root, env=env, capture_output=True, text=True, check=False
    )
    return done.returncode == 0, done.stdout[-1500:] + done.stderr[-500:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/mutants.txt"))
    args = parser.parse_args()
    repo = args.repo.resolve()

    rows, survivors = [], []
    with tempfile.TemporaryDirectory(prefix="vlashift-mutants-") as tmp:
        base = Path(tmp) / "base"
        for name in ("src", "tests", "scripts"):
            shutil.copytree(repo / name, base / name, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(repo / "pyproject.toml", base / "pyproject.toml")
        passed, output = run_tests(base)
        if not passed:
            print("the unmodified copy fails its own tests, nothing to compare with")
            print(output)
            return 2
        for number, (name, rel, old, new) in enumerate(MUTANTS):
            root = Path(tmp) / f"mutant{number}"
            shutil.copytree(base, root)
            apply(root / rel, old, new)
            killed = not run_tests(root)[0]
            rows.append(f"{'killed' if killed else 'SURVIVED':9} {name:34} {rel}")
            print(rows[-1], flush=True)
            if not killed:
                survivors.append(name)
            shutil.rmtree(root)

    summary = f"{len(MUTANTS) - len(survivors)} of {len(MUTANTS)} mutants killed"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join([summary, *rows]) + "\n", encoding="utf-8")
    print(summary)
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
