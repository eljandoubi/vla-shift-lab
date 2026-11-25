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
command -v uv > /dev/null || pip -q install uv
uv venv --python 3.12 "$LR"
uv pip install --python "$PY" "lerobot[smolvla,libero]"
uv pip install --python "$PY" -e "$REPO_DIR[report]"

# hf-libero asks a question with input() at the first import when ~/.libero/config.yaml is missing
mkdir -p "$HOME/.libero"
[ -f "$HOME/.libero/config.yaml" ] || echo N | "$PY" -c "import libero.libero"

"$PY" -c "import torch; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available())"
"$PY" - << 'EOF'
import mujoco

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
