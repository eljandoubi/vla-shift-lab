"""The arms of the study and the environment variables that configure the patched script."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from vlashift.instructions import MODES

# Levels per shift, the main level first. Brightness is a multiplier, noise a standard deviation
# on pixels in [0, 1], occlusion the side of a centred square as a fraction of the image side,
# joint a standard deviation in radians on each of the 7 arm joints.
SHIFT_LEVELS = {
    "bright": (0.5, 1.5),
    "noise": (0.1, 0.05),
    "occl": (0.4, 0.2),
    "joint": (0.1, 0.05, 0.2),
}
SHIFT_ENV = {
    "bright": "VLASHIFT_BRIGHTNESS",
    "noise": "VLASHIFT_NOISE_STD",
    "occl": "VLASHIFT_OCCLUSION",
    "joint": "VLASHIFT_JOINT_STD",
}
INSTRUCTION_ENV = "VLASHIFT_INSTRUCTION"
SEED_ENV = "VLASHIFT_SEED"
CAMERAS_ENV = "VLASHIFT_CAMERAS"
VIDEOS_ENV = "VLASHIFT_VIDEOS"
FRAMES_ENV = "VLASHIFT_DUMP_FRAMES"
APPLIED_ENV = "VLASHIFT_APPLIED_PATH"
KNOWN_ENV = frozenset(
    {
        INSTRUCTION_ENV,
        SEED_ENV,
        CAMERAS_ENV,
        VIDEOS_ENV,
        FRAMES_ENV,
        APPLIED_ENV,
        *SHIFT_ENV.values(),
    }
)

STAGES = {
    "baseline": ("keep",),
    "instruction": ("blank", "other", "shuffled", "paraphrase"),
    "shifts": tuple(f"{kind}_{levels[0]:g}" for kind, levels in SHIFT_LEVELS.items()),
    "shifts_extra": tuple(
        f"{kind}_{level:g}" for kind, levels in SHIFT_LEVELS.items() for level in levels[1:]
    ),
}
ALL_ARMS = (
    *MODES,
    *(f"{kind}_{level:g}" for kind, levels in SHIFT_LEVELS.items() for level in sorted(levels)),
)


@dataclass(frozen=True)
class Arm:
    name: str
    instruction: str = "keep"
    shift: str | None = None
    level: float | None = None

    def env(self) -> dict[str, str]:
        env = {INSTRUCTION_ENV: self.instruction}
        if self.shift is not None:
            env[SHIFT_ENV[self.shift]] = f"{self.level:g}"
        return env


def parse_arm(name: str) -> Arm:
    if name in MODES:
        return Arm(name, instruction=name)
    kind, _, text = name.partition("_")
    if kind not in SHIFT_LEVELS:
        raise ValueError(
            f"unknown arm {name!r}: use one of {MODES} or a shift {tuple(SHIFT_LEVELS)} "
            "followed by a level, for example noise_0.1"
        )
    try:
        level = float(text)
    except ValueError:
        raise ValueError(f"arm {name!r} needs a numeric level after the underscore") from None
    if not level > 0:
        raise ValueError(f"arm {name!r} needs a positive level")
    return Arm(f"{kind}_{level:g}", shift=kind, level=level)


def shell_lines(arm: Arm) -> list[str]:
    """Shell code that clears every study variable, then exports the ones of the arm."""
    cleared = [f"unset {name}" for name in sorted(KNOWN_ENV)]
    exported = [f"export {name}={value}" for name, value in sorted(arm.env().items())]
    return cleared + exported


def _number(environ: Mapping[str, str], name: str, cast, check, expected: str):
    if name not in environ:
        return None
    try:
        value = cast(environ[name])
    except ValueError:
        raise ValueError(f"{name}={environ[name]!r} is not a number") from None
    if not check(value):
        raise ValueError(f"{name}={environ[name]!r} must be {expected}")
    return value


@dataclass(frozen=True)
class RuntimeConfig:
    instruction: str = "keep"
    brightness: float | None = None
    noise_std: float | None = None
    occlusion: float | None = None
    joint_std: float | None = None
    cameras: tuple[str, ...] | None = None
    seed: int = 0
    video_episodes: int = 0
    dump_frames: int = 0
    applied_path: str | None = None

    @property
    def image_shift(self) -> bool:
        return any(v is not None for v in (self.brightness, self.noise_std, self.occlusion))

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> Self:
        environ = os.environ if environ is None else environ
        unknown = sorted(k for k in environ if k.startswith("VLASHIFT_") and k not in KNOWN_ENV)
        if unknown:
            raise ValueError(f"unknown variable(s) {unknown}, known ones: {sorted(KNOWN_ENV)}")
        instruction = environ.get(INSTRUCTION_ENV, "keep")
        if instruction not in MODES:
            raise ValueError(f"{INSTRUCTION_ENV}={instruction!r} is not one of {MODES}")
        cameras = environ.get(CAMERAS_ENV)
        names = tuple(part.strip() for part in cameras.split(",")) if cameras else None
        if names is not None and not all(names):
            raise ValueError(f"{CAMERAS_ENV}={cameras!r} has an empty camera name")
        non_negative = lambda v: v >= 0  # noqa: E731
        return cls(
            instruction=instruction,
            brightness=_number(environ, SHIFT_ENV["bright"], float, lambda v: v > 0, "positive"),
            noise_std=_number(environ, SHIFT_ENV["noise"], float, non_negative, "not negative"),
            occlusion=_number(
                environ, SHIFT_ENV["occl"], float, lambda v: 0 <= v <= 1, "between 0 and 1"
            ),
            joint_std=_number(environ, SHIFT_ENV["joint"], float, non_negative, "not negative"),
            cameras=names,
            seed=_number(environ, SEED_ENV, int, lambda v: True, "an integer") or 0,
            video_episodes=_number(environ, VIDEOS_ENV, int, non_negative, "not negative") or 0,
            dump_frames=_number(environ, FRAMES_ENV, int, non_negative, "not negative") or 0,
            applied_path=environ.get(APPLIED_ENV),
        )
