import json
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import flags, uniform, write_run

from vlashift.report import (
    cell,
    delta_points,
    episode_csv,
    episodes_in,
    figures,
    format_p,
    h1_verdict,
    h2_verdict,
    h3_verdict,
    hypothesis_rows,
    hypothesis_table,
    ordered_arms,
    paired_rows,
    paired_table,
    results_table,
    timing_summary,
    update_readme,
    write_all,
)
from vlashift.results import load_tree

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tree(tmp_path):
    for suite in ("libero_spatial", "libero_goal"):
        write_run(tmp_path, "keep", suite, uniform(9), eval_ep_s=8.0)
        write_run(tmp_path, "blank", suite, uniform(1), eval_ep_s=6.0)
    write_run(tmp_path, "noise_0.1", "libero_goal", {t: flags(9) for t in range(10)})
    write_run(tmp_path, "mystery", "libero_goal", uniform(5))
    return load_tree(tmp_path)


def test_ordered_arms_follow_the_canonical_order_then_unknown_names(tree):
    assert ordered_arms(tree) == ["keep", "blank", "noise_0.1", "mystery"]


def test_episodes_in_a_suite_or_everywhere(tree):
    one = episodes_in(tree["keep"], "libero_goal")
    assert len(one) == 100
    assert ("libero_goal", 3, 9) in one
    assert len(episodes_in(tree["keep"], "all")) == 200
    assert episodes_in(tree["noise_0.1"], "libero_spatial") == {}


def test_episode_csv_has_one_line_per_episode_in_arm_and_suite_order(tree):
    lines = episode_csv(tree).splitlines()
    assert lines[0] == "arm,suite,task,episode,success"
    assert len(lines) == 601
    assert lines[1] == "keep,libero_spatial,0,0,1"
    assert "keep,libero_spatial,0,9,0" in lines
    assert lines[101] == "keep,libero_goal,0,0,1"
    assert lines[-1] == "mystery,libero_goal,9,9,0"


def test_cell_text():
    assert cell(90, 100) == "90.0 [82.6, 94.5] (90/100)"
    assert cell(0, 0) == "not run"


def test_results_table_has_one_row_per_arm_and_marks_missing_cells(tree):
    lines = results_table(tree).splitlines()
    assert lines[0] == "| arm | Spatial | Object | Goal | Long | All |"
    assert [line.split("|")[1].strip() for line in lines[2:]] == [
        "keep",
        "blank",
        "noise_0.1",
        "mystery",
    ]
    keep = lines[2].split("|")
    assert keep[1].strip() == "keep"
    assert keep[2].strip() == "90.0 [82.6, 94.5] (90/100)"
    assert keep[3].strip() == "not run"
    assert keep[5].strip() == "not run"
    assert keep[6].strip() == "90.0 [85.1, 93.4] (180/200)"


def test_paired_rows_count_discordant_episodes_and_take_the_difference(tree):
    rows = {(r["arm"], r["scope"]): r for r in paired_rows(tree, n_boot=500)}
    blank = rows[("blank", "libero_spatial")]
    assert blank["pairs"] == 100
    assert (blank["keep_only"], blank["arm_only"]) == (80, 0)
    assert blank["delta"] == pytest.approx(80.0)
    assert (blank["low"], blank["high"]) == (pytest.approx(80.0), pytest.approx(80.0))
    assert blank["p"] < 1e-20
    assert rows[("blank", "all")]["pairs"] == 200
    noise = rows[("noise_0.1", "libero_goal")]
    assert (noise["keep_only"], noise["arm_only"], noise["delta"]) == (0, 0, 0.0)
    assert noise["p"] == 1.0


def test_paired_rows_skip_scopes_without_common_episodes(tree):
    scopes = {r["scope"] for r in paired_rows(tree, n_boot=100) if r["arm"] == "noise_0.1"}
    assert scopes == {"libero_goal", "all"}


def test_paired_rows_need_the_baseline(tmp_path):
    write_run(tmp_path, "blank", "libero_goal", uniform(1))
    assert paired_rows(load_tree(tmp_path)) == []
    assert "No paired comparison" in paired_table([])


def test_paired_table_formats_each_row(tree):
    text = paired_table(paired_rows(tree, n_boot=200))
    header, rule, first = text.splitlines()[:3]
    assert "keep minus arm (points)" in header
    assert "uncorrected" in header
    assert rule.count("---") == 7
    assert first.startswith("| blank | Spatial | 100 | +80.0 | [+80.0, +80.0] | 80 / 0 | <0.001 |")


@pytest.mark.parametrize(("p", "text"), [(0.0004, "<0.001"), (0.0123, "0.012"), (1.0, "1.000")])
def test_format_p(p, text):
    assert format_p(p) == text


def test_delta_points_is_keep_minus_arm_on_the_shared_episodes(tree):
    assert delta_points(tree, "blank", "libero_spatial") == pytest.approx(80.0)
    assert delta_points(tree, "blank", "all") == pytest.approx(80.0)
    assert delta_points(tree, "noise_0.1", "libero_spatial") is None
    assert delta_points(tree, "missing", "all") is None


@pytest.mark.parametrize(
    ("spatial", "obj", "goal", "verdict"),
    [
        (0, 0, 40, "holds"),
        (14, 0, 29, "holds"),
        (0, 14, 29, "holds"),
        (14, 0, 28, "unclear"),
        (15, 0, 60, "unclear"),
        (0, 15, 60, "unclear"),
        (29, 0, 90, "unclear"),
        (30, 0, 90, "falsified"),
        (0, 30, 90, "falsified"),
        (5, 5, 5, "falsified"),
        (5, 3, 5, "falsified"),
        (3, 5, 5, "falsified"),
        (5, 3, 6, "unclear"),
    ],
)
def test_h1_verdict_follows_the_thresholds_of_the_readme(spatial, obj, goal, verdict):
    assert h1_verdict(spatial, obj, goal) == verdict


def test_h1_without_a_goal_run_can_only_be_falsified() -> None:
    assert h1_verdict(30, 0) == "falsified"
    assert h1_verdict(0, 30) == "falsified"
    assert h1_verdict(29, 0) == "not run"
    assert h1_verdict(0, 0) == "not run"


@pytest.mark.parametrize(
    ("spatial", "obj", "verdict"),
    [
        (0, 0, "holds"),
        (9, 0, "holds"),
        (0, 9, "holds"),
        (-5, -3, "holds"),
        (10, 0, "unclear"),
        (0, 10, "unclear"),
        (19, 0, "unclear"),
        (20, 0, "falsified"),
        (0, 20, "falsified"),
        (20, None, "falsified"),
        (None, 20, "falsified"),
        (19, None, "not run"),
        (None, 0, "not run"),
        (None, None, "not run"),
    ],
)
def test_h2_verdict_follows_the_thresholds_of_the_readme(spatial, obj, verdict):
    assert h2_verdict(spatial, obj) == verdict


@pytest.mark.parametrize(
    ("noise", "occl", "bright", "verdict"),
    [
        ((20, 10, 30), (25, 12, 35), (2, -5, 8), "holds"),
        ((20, 10, 30), (25, 12, 35), (2, -5, 11), "unclear"),
        ((20, 10, 30), (25, 8, 35), (2, -5, 8), "unclear"),
        ((20, 8, 30), (25, 12, 35), (2, -5, 8), "unclear"),
        ((20, 10, 30), (25, 12, 35), (20, 10, 30), "falsified"),
        ((20, 10, 30), (25, 12, 35), (25, 18, 32), "falsified"),
        ((20, 10, 30), (25, 12, 35), (-4, -9, 1), "holds"),
    ],
)
def test_h3_verdict_needs_both_intervals_above_the_brightness_interval(
    noise, occl, bright, verdict
):
    assert h3_verdict(noise, occl, bright) == verdict


def test_h3_needs_the_intervals_to_clear_each_other_not_just_touch():
    assert h3_verdict((20, 8, 30), (25, 8, 35), (2, -5, 8)) == "unclear"
    assert h3_verdict((20, 9, 30), (25, 9, 35), (2, -5, 8)) == "holds"


def test_hypothesis_rows_apply_the_rules_to_a_complete_tree(tmp_path):
    suites = ("libero_spatial", "libero_object", "libero_goal")
    for suite in suites:
        write_run(tmp_path, "keep", suite, uniform(9))
    for suite, kept in zip(suites, (8, 9, 2), strict=True):
        write_run(tmp_path, "blank", suite, uniform(kept))
    for suite in suites[:2]:
        write_run(tmp_path, "other", suite, uniform(9))
    for arm, kept in (("noise_0.1", 3), ("occl_0.4", 5), ("bright_0.5", 9)):
        for suite in suites:
            write_run(tmp_path, arm, suite, uniform(kept))
    rows = hypothesis_rows(load_tree(tmp_path))
    assert [row[2] for row in rows] == ["holds", "holds", "holds"]
    assert rows[0][1] == "blank: Spatial +10.0, Object +0.0, Goal +70.0"
    assert rows[1][1] == "other: Spatial +0.0, Object +0.0"
    assert rows[2][1] == (
        "noise_0.1 +60.0 [+60.0, +60.0], occl_0.4 +40.0 [+40.0, +40.0], "
        "bright_0.5 +0.0 [+0.0, +0.0]"
    )


def test_hypothesis_rows_falsify_h1_without_a_goal_run(tmp_path):
    for suite in ("libero_spatial", "libero_object"):
        write_run(tmp_path, "keep", suite, uniform(9))
        write_run(tmp_path, "blank", suite, uniform(1))
    rows = hypothesis_rows(load_tree(tmp_path))
    assert rows[0][1] == "blank: Spatial +80.0, Object +80.0"
    assert rows[0][2] == "falsified"


def test_hypothesis_rows_falsify_h2_from_one_suite(tmp_path):
    write_run(tmp_path, "keep", "libero_spatial", uniform(9))
    write_run(tmp_path, "other", "libero_spatial", uniform(2))
    rows = hypothesis_rows(load_tree(tmp_path))
    assert rows[1][1] == "other: Spatial +70.0"
    assert rows[1][2] == "falsified"


def test_hypothesis_rows_wait_for_both_suites_unless_h2_is_falsified(tmp_path):
    write_run(tmp_path, "keep", "libero_spatial", uniform(9))
    write_run(tmp_path, "other", "libero_spatial", uniform(9))
    assert hypothesis_rows(load_tree(tmp_path))[1][2] == "not run"


def test_hypothesis_rows_say_not_run_when_an_arm_is_missing(tree):
    rows = hypothesis_rows(tree)
    assert [row[2] for row in rows] == ["not run", "not run", "not run"]
    assert rows[0][1] == "blank: Spatial +80.0, Goal +80.0"
    assert rows[1][1] == ""
    assert rows[2][1] == "noise_0.1 +0.0 [+0.0, +0.0]"


def test_h3_waits_for_the_same_episodes_in_every_arm(tmp_path):
    suites = ("libero_spatial", "libero_object", "libero_goal")
    for suite in suites:
        write_run(tmp_path, "keep", suite, uniform(9))
    for arm, kept in (("noise_0.1", 3), ("occl_0.4", 5), ("bright_0.5", 9)):
        for suite in suites if arm != "bright_0.5" else suites[:2]:
            write_run(tmp_path, arm, suite, uniform(kept))
    assert hypothesis_rows(load_tree(tmp_path))[2][2] == "not run"


def test_hypothesis_table_has_one_row_per_hypothesis(tree):
    lines = hypothesis_table(hypothesis_rows(tree)).splitlines()
    assert lines[0] == "| hypothesis | points, keep minus arm | verdict |"
    assert [line.split(" ")[1] for line in lines[2:]] == ["H1:", "H2:", "H3:"]


def test_timing_summary_weights_runs_by_episodes(tree):
    text = timing_summary(tree)
    assert text.startswith("6 runs, 600 episodes")
    assert "7.2 s per episode" in text
    assert "1.2 h in total" in text


def test_timing_summary_without_timing(tmp_path):
    directory = write_run(tmp_path, "keep", "libero_goal", uniform(9))
    info = json.loads((directory / "eval_info.json").read_text())
    del info["overall"]["eval_ep_s"]
    (directory / "eval_info.json").write_text(json.dumps(info))
    assert timing_summary(load_tree(tmp_path)) == "No timing in the runs."


def test_write_all_creates_tables_summary_and_figures(tmp_path):
    runs = tmp_path / "runs"
    for suite in ("libero_spatial", "libero_goal"):
        write_run(runs, "keep", suite, uniform(9))
        write_run(runs, "blank", suite, uniform(1))
    (runs / "blank" / "libero_object.partial").mkdir()
    out = tmp_path / "out"
    table = write_all(runs, out)
    text = (out / "tables.md").read_text()
    assert table.startswith(results_table(load_tree(runs)))
    assert "| H1:" in table
    assert "## Hypotheses" in text
    assert "## Paired comparison with keep" in text
    assert "blank/libero_object.partial" in text
    summary = json.loads((out / "summary.json").read_text())
    assert summary["cells"]["keep"]["libero_goal"] == {"successes": 90, "episodes": 100}
    assert {r["arm"] for r in summary["paired"]} == {"blank"}
    assert (out / "figures" / "success_by_arm.png").stat().st_size > 1000
    assert (out / "figures" / "drop_vs_keep.png").stat().st_size > 1000
    lines = (out / "episodes.csv").read_text().splitlines()
    assert lines[0] == "arm,suite,task,episode,success"
    assert len(lines) == 401


def test_figures_without_a_baseline_only_draw_the_first_chart(tmp_path):
    write_run(tmp_path / "runs", "blank", "libero_goal", uniform(1))
    figures(load_tree(tmp_path / "runs"), tmp_path / "fig")
    assert (tmp_path / "fig" / "success_by_arm.png").exists()
    assert not (tmp_path / "fig" / "drop_vs_keep.png").exists()


def test_write_all_refuses_an_empty_tree(tmp_path):
    with pytest.raises(ValueError, match="no finished run"):
        write_all(tmp_path, tmp_path / "out")


def test_update_readme_replaces_only_the_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- results:start -->\nold\n<!-- results:end -->\noutro\n")
    update_readme(readme, "| new |")
    assert (
        readme.read_text()
        == "intro\n<!-- results:start -->\n| new |\n<!-- results:end -->\noutro\n"
    )


@pytest.mark.parametrize(
    "text", ["no markers", "<!-- results:start -->\n<!-- results:start -->\n<!-- results:end -->"]
)
def test_update_readme_needs_exactly_one_pair_of_markers(tmp_path, text):
    readme = tmp_path / "README.md"
    readme.write_text(text)
    with pytest.raises(ValueError, match="marker"):
        update_readme(readme, "x")


def test_make_tables_script_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "keep", "libero_goal", uniform(9))
    write_run(runs, "other", "libero_goal", uniform(2))
    readme = tmp_path / "README.md"
    readme.write_text("<!-- results:start -->\n<!-- results:end -->\n")
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_tables.py"), "--runs", str(runs),
         "--out", str(tmp_path / "res"), "--readme", str(readme)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "| other |" in readme.read_text()
    assert (tmp_path / "res" / "tables.md").exists()
