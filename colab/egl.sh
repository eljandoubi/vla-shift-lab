#!/usr/bin/env bash
# usage: egl.sh
# The Colab image ships NVIDIA's EGL library but no EGL vendor file for it, so EGL falls back to
# Mesa's software renderer: a LIBERO step takes about 0.5 s instead of 35 ms and one episode
# takes minutes. This writes the vendor file. Run as root, once per runtime.
set -euo pipefail
LIB=${NVIDIA_EGL_LIB:-/usr/lib64-nvidia/libEGL_nvidia.so.0}
DIR=${EGL_VENDOR_DIR:-/usr/share/glvnd/egl_vendor.d}

if [ ! -e "$LIB" ]; then
  echo "no NVIDIA EGL library at $LIB, rendering stays on the CPU"
  exit 0
fi
mkdir -p "$DIR"
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s"}}\n' "$LIB" > "$DIR/10_nvidia.json"
echo "EGL vendor file written to $DIR/10_nvidia.json"
