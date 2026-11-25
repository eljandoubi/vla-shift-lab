import json

import pytest

from vlashift.checks import applied_problems, description_problems, read_applied
from vlashift.config import ALL_ARMS, RuntimeConfig, parse_arm
from vlashift.suites import instructions

IMAGE = {"bright", "noise", "occl"}


def record(arm_name, changed=0, images=0, states=0, first=None):
    config = RuntimeConfig.from_environ(parse_arm(arm_name).env())
    return {
        "config": {
            "instruction": config.instruction,
            "brightness": config.brightness,
            "noise_std": config.noise_std,
            "occlusion": config.occlusion,
            "joint_std": config.joint_std,
        },
        "counts": {
            "instructions_changed": changed,
            "images_perturbed": images,
            "init_states_perturbed": states,
        },
        "first_states": first if first is not None else [[0.1, 0.2, 0.3] + [0.0] * 5],
    }


def good(arm_name):
    arm = parse_arm(arm_name)
    return record(
        arm_name,
        changed=40 if arm.instruction != "keep" else 0,
        images=300 if arm.shift in IMAGE else 0,
        states=50 if arm.shift == "joint" else 0,
    )


@pytest.mark.parametrize("name", ALL_ARMS)
def test_a_record_that_matches_the_arm_has_no_problem(name):
    assert applied_problems(parse_arm(name), good(name)) == []


def test_a_baseline_with_leaked_shift_is_flagged():
    problems = applied_problems(parse_arm("keep"), record("keep", images=5, states=3, changed=2))
    assert len(problems) == 3


def test_a_leaked_variable_is_flagged_by_the_setting():
    applied = good("keep")
    applied["config"]["noise_std"] = 0.1
    assert "noise_std" in applied_problems(parse_arm("keep"), applied)[0]


@pytest.mark.parametrize("name", ["blank", "other", "shuffled", "paraphrase"])
def test_an_instruction_arm_that_changed_nothing_is_flagged(name):
    assert applied_problems(parse_arm(name), record(name)) == ["the instruction was never changed"]


@pytest.mark.parametrize("name", ["bright_0.5", "noise_0.1", "occl_0.4"])
def test_an_image_arm_without_perturbed_images_is_flagged(name):
    assert applied_problems(parse_arm(name), record(name)) == ["no image was perturbed"]


def test_a_joint_arm_without_perturbed_states_is_flagged():
    problems = applied_problems(parse_arm("joint_0.1"), record("joint_0.1"))
    assert len(problems) == 1
    assert "worker processes" in problems[0]


def test_a_joint_arm_must_move_the_start_pose_against_the_baseline():
    arm = parse_arm("joint_0.2")
    base = good("keep")
    moved = good("joint_0.2")
    moved["first_states"] = [[0.12, 0.2, 0.3] + [0.0] * 5]
    assert applied_problems(arm, moved, baseline=base) == []
    same = good("joint_0.2")
    assert "no visible effect" in applied_problems(arm, same, baseline=base)[0]
    nothing = good("joint_0.2")
    nothing["first_states"] = []
    assert "no first state" in applied_problems(arm, nothing, baseline=base)[0]


def test_baseline_comparison_is_only_for_joint_arms():
    assert applied_problems(parse_arm("noise_0.1"), good("noise_0.1"), baseline=good("keep")) == []


def test_read_applied_loads_the_json(tmp_path):
    (tmp_path / "vlashift_applied.json").write_text(json.dumps(good("keep")))
    assert read_applied(tmp_path)["counts"]["images_perturbed"] == 0


def write_descriptions(tmp_path, mapping):
    (tmp_path / "task_descriptions.json").write_text(json.dumps(mapping))


def test_descriptions_equal_to_the_table_pass(tmp_path):
    write_descriptions(
        tmp_path, {f"libero_goal_{i}": t for i, t in enumerate(instructions()["libero_goal"])}
    )
    assert description_problems(tmp_path) == []


def test_task_names_ending_with_the_text_pass(tmp_path):
    text = instructions()["libero_10"][2]
    name = "KITCHEN_SCENE3_" + text.replace(" ", "_")
    write_descriptions(tmp_path, {"libero_10_2": name})
    assert description_problems(tmp_path) == []


def test_a_different_text_is_reported(tmp_path):
    write_descriptions(tmp_path, {"libero_goal_0": "open the top drawer"})
    problems = description_problems(tmp_path)
    assert len(problems) == 1
    assert "libero_goal_0" in problems[0]


def test_unknown_keys_and_missing_files_are_reported(tmp_path):
    assert description_problems(tmp_path) == ["task_descriptions.json is missing"]
    write_descriptions(tmp_path, {"libero_90_3": "x", "libero_goal_12": "y", "oops": "z"})
    assert len(description_problems(tmp_path)) == 3
