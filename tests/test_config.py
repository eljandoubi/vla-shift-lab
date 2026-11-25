import pytest

from vlashift.config import (
    ALL_ARMS,
    KNOWN_ENV,
    SHIFT_ENV,
    SHIFT_LEVELS,
    STAGES,
    Arm,
    RuntimeConfig,
    parse_arm,
    shell_lines,
)
from vlashift.instructions import MODES


def test_there_are_fourteen_arms_with_unique_names():
    assert len(ALL_ARMS) == 14
    assert len(set(ALL_ARMS)) == 14
    assert ALL_ARMS[:5] == MODES


def test_stages_partition_the_arms():
    assert [len(STAGES[s]) for s in ("baseline", "instruction", "shifts", "shifts_extra")] == [
        1,
        4,
        4,
        5,
    ]
    names = [arm for arms in STAGES.values() for arm in arms]
    assert sorted(names) == sorted(ALL_ARMS)


def test_main_shift_level_comes_first_in_the_shifts_stage():
    assert STAGES["shifts"] == ("bright_0.5", "noise_0.1", "occl_0.4", "joint_0.1")
    assert "joint_0.2" in STAGES["shifts_extra"]
    assert "joint_0.1" not in STAGES["shifts_extra"]


@pytest.mark.parametrize("name", ALL_ARMS)
def test_parse_arm_round_trip(name):
    assert parse_arm(name).name == name


@pytest.mark.parametrize("name", MODES)
def test_instruction_arm_sets_only_the_instruction(name):
    assert parse_arm(name).env() == {"VLASHIFT_INSTRUCTION": name}


@pytest.mark.parametrize(
    ("kind", "level"), [(k, lv) for k, levels in SHIFT_LEVELS.items() for lv in levels]
)
def test_shift_arm_sets_the_instruction_and_one_shift(kind, level):
    arm = parse_arm(f"{kind}_{level:g}")
    assert arm.env() == {"VLASHIFT_INSTRUCTION": "keep", SHIFT_ENV[kind]: f"{level:g}"}
    assert (arm.shift, arm.level) == (kind, level)


@pytest.mark.parametrize(
    "name", ["swap", "noise", "noise_", "noise_abc", "noise_-1", "noise_0", "fog_0.5"]
)
def test_parse_arm_rejects_bad_names(name):
    with pytest.raises(ValueError):
        parse_arm(name)


def test_parse_arm_normalises_the_level_text():
    assert parse_arm("noise_0.10").name == "noise_0.1"


def test_shell_lines_clear_everything_before_exporting():
    lines = shell_lines(parse_arm("noise_0.05"))
    unsets = [ln for ln in lines if ln.startswith("unset ")]
    exports = [ln for ln in lines if ln.startswith("export ")]
    assert {ln.split()[1] for ln in unsets} == set(KNOWN_ENV)
    assert exports == ["export VLASHIFT_INSTRUCTION=keep", "export VLASHIFT_NOISE_STD=0.05"]
    assert lines.index(unsets[-1]) < lines.index(exports[0])


@pytest.mark.parametrize("name", ALL_ARMS)
def test_runtime_config_reads_back_the_arm(name):
    arm = parse_arm(name)
    config = RuntimeConfig.from_environ(arm.env())
    assert config.instruction == arm.instruction
    field = {
        "bright": "brightness",
        "noise": "noise_std",
        "occl": "occlusion",
        "joint": "joint_std",
    }
    for kind, attribute in field.items():
        expected = arm.level if arm.shift == kind else None
        assert getattr(config, attribute) == expected


def test_runtime_config_defaults():
    config = RuntimeConfig.from_environ({})
    assert config == RuntimeConfig()
    assert not config.image_shift
    assert (config.seed, config.video_episodes, config.dump_frames) == (0, 0, 0)


def test_runtime_config_reads_every_option():
    config = RuntimeConfig.from_environ(
        {
            "VLASHIFT_SEED": "7",
            "VLASHIFT_CAMERAS": "image, image2",
            "VLASHIFT_VIDEOS": "2",
            "VLASHIFT_DUMP_FRAMES": "3",
            "VLASHIFT_APPLIED_PATH": "/tmp/x.json",
            "VLASHIFT_OCCLUSION": "0.4",
        }
    )
    assert config.seed == 7
    assert config.cameras == ("image", "image2")
    assert (config.video_episodes, config.dump_frames) == (2, 3)
    assert config.applied_path == "/tmp/x.json"
    assert config.image_shift


def test_runtime_config_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("VLASHIFT_JOINT_STD", "0.2")
    assert RuntimeConfig.from_environ().joint_std == 0.2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VLASHIFT_INSTRUCTION", "swap"),
        ("VLASHIFT_BRIGHTNESS", "0"),
        ("VLASHIFT_BRIGHTNESS", "dark"),
        ("VLASHIFT_NOISE_STD", "-0.1"),
        ("VLASHIFT_NOISE_STD", ""),
        ("VLASHIFT_OCCLUSION", "1.5"),
        ("VLASHIFT_JOINT_STD", "-1"),
        ("VLASHIFT_SEED", "1.5"),
        ("VLASHIFT_VIDEOS", "-1"),
        ("VLASHIFT_CAMERAS", "image,"),
    ],
)
def test_runtime_config_rejects_bad_values(name, value):
    with pytest.raises(ValueError):
        RuntimeConfig.from_environ({name: value})


def test_runtime_config_rejects_misspelled_variables():
    with pytest.raises(ValueError, match="unknown variable"):
        RuntimeConfig.from_environ({"VLASHIFT_NOISE": "0.1"})


def test_runtime_config_ignores_foreign_variables():
    assert RuntimeConfig.from_environ({"HOME": "/root", "OTHER_NOISE": "x"}) == RuntimeConfig()


def test_arm_is_hashable_and_frozen():
    arm = Arm("keep")
    assert {arm: 1}[Arm("keep")] == 1
    with pytest.raises(AttributeError):
        arm.name = "blank"
