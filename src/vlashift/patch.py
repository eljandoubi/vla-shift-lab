"""Build the patched copy of lerobot's evaluation script that runs the arms.

The copy differs from the original in five places. Each one is located by an exact string that
must occur once; anything else stops the build, so a LeRobot update cannot slip through with a
hook silently missing.
"""

import hashlib
import importlib.util
import os
from pathlib import Path

ANCHORS = (
    (
        "hook import",
        "logger = logging.getLogger(__name__)\n",
        "logger = logging.getLogger(__name__)\n"
        "import vlashift.runtime as _vlashift  # noqa: E402\n",
    ),
    (
        "instruction",
        'observation["task"] = list(env.call("task_description"))',
        'observation["task"] = _vlashift.instructions(list(env.call("task_description")))',
    ),
    (
        "images",
        "observation = env_preprocessor(observation)",
        "observation = _vlashift.images(env_preprocessor(observation))",
    ),
    (
        "videos",
        "max_episodes_rendered = 0 if cfg.eval.recording else 10",
        "max_episodes_rendered = 0 if cfg.eval.recording else _vlashift.video_episodes()",
    ),
    (
        "install",
        "    register_third_party_plugins()\n    eval_main()",
        "    register_third_party_plugins()\n    _vlashift.install()\n    eval_main()",
    ),
)

HEADER = (
    "# Patched copy of lerobot/scripts/lerobot_eval.py (sha256 of the original: {digest}).\n"
    "# Built by vlashift.patch; the hooks read their settings from VLASHIFT_* variables.\n"
)


class AnchorError(RuntimeError):
    pass


def build(source: str) -> str:
    patched = source
    for name, old, new in ANCHORS:
        count = patched.count(old)
        if count != 1:
            raise AnchorError(
                f"anchor {name!r} occurs {count} times in the eval script, expected once: {old!r}"
            )
        patched = patched.replace(old, new)
    digest = hashlib.sha256(source.encode()).hexdigest()
    return HEADER.format(digest=digest) + patched


def lerobot_dir() -> Path:
    """The LeRobot package directory: LEROBOT_SRC, else the installed package."""
    given = os.environ.get("LEROBOT_SRC")
    if given:
        return Path(given)
    spec = importlib.util.find_spec("lerobot")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(
            "LeRobot is not installed, set LEROBOT_SRC to its package directory"
        )
    return Path(next(iter(spec.submodule_search_locations)))


def eval_script() -> Path:
    return lerobot_dir() / "scripts" / "lerobot_eval.py"


def build_file(out: Path, source: Path | None = None) -> str:
    """Write the patched script to out after checking that it compiles; return its text."""
    text = build((source or eval_script()).read_text(encoding="utf-8"))
    compile(text, str(out), "exec")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text
