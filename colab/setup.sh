#!/usr/bin/env bash
# usage: setup.sh
# One-time setup on a Colab GPU runtime: system libraries, a Python 3.12 virtualenv with LeRobot
# and LIBERO, this package, the patched evaluation script and a rendering check. A few minutes.
# Clone the repository to $REPO_DIR first (git clone <url> /content/vla-shift-lab).
set -euo pipefail
. "$(dirname "$0")/common.sh"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apt-get -qq update
apt-get -qq install -y ffmpeg git libegl1 libgl1 libglvnd0 libosmesa6 > /dev/null
bash "$HERE/egl.sh"
command -v uv > /dev/null || pip -q install uv
uv venv --python 3.12 "$LR"
uv pip install --python "$PY" "lerobot[smolvla,libero]"
# MuJoCo 3.4 and later break the start state of libero_spatial task 5 (lerobot issue 4390).
uv pip install --python "$PY" "mujoco==3.3.7"
"$PY" -c "import mujoco; print('mujoco', mujoco.__version__)"
uv pip install --python "$PY" -e "$REPO_DIR[report]"

# hf-libero asks a question with input() at the first import when ~/.libero/config.yaml is missing
mkdir -p "$HOME/.libero"
[ -f "$HOME/.libero/config.yaml" ] || echo N | "$PY" -c "import libero.libero"

"$PY" -c "import torch; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available())"
"$PY" - << 'EOF'
import mujoco
from mujoco.egl import GLContext
from OpenGL import GL

context = GLContext(max_width=64, max_height=64)
context.make_current()
name = GL.glGetString(GL.GL_RENDERER).decode()
print("OpenGL renderer:", name)
if "llvmpipe" in name:
    print("WARNING: software rendering, an episode will take minutes instead of seconds")

model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, 64, 64)
renderer.update_scene(data)
print("offscreen rendering ok", renderer.render().shape)
EOF

mkdir -p "$OUT"
"$PY" -m vlashift.cli build-eval --out "$PATCHED"
"$PY" -m py_compile "$PATCHED"
echo "setup done: patched script at $PATCHED, results go to $OUT"
