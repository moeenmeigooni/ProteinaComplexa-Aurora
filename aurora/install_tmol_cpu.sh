#!/usr/bin/env bash
# Added for the Aurora XPU port; see aurora/README.md.
# Build TMol without CUDA for Aurora's CPU fallback path.
#
# TMol has no Intel XPU backend.  A CPU-only build preserves its interface
# energy / gradient functionality while the main Proteina models run on XPU.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 VENV_PREFIX" >&2
  exit 2
fi

venv_prefix="$1"
python_bin="${venv_prefix}/bin/python"

"${python_bin}" -m pip install --upgrade scikit-build-core
if ! "${python_bin}" -c 'import pybind11' >/dev/null 2>&1; then
  "${python_bin}" -m pip install --no-deps pybind11
fi

# Resolve TMol's own pure-Python and CPU dependencies explicitly, keeping
# pip's resolver away from the vendor XPU PyTorch distribution.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${python_bin}" -m pip install --no-deps -r "${script_dir}/requirements-tmol-cpu.txt"

# Reuse the ABI-compatible vendor Torch and the framework's pybind11 CMake
# package.  Do not fetch TMol's prebuilt CUDA wheel.
torch_cmake_prefix="$("${python_bin}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
pybind11_cmake_prefix="$("${python_bin}" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
export CMAKE_PREFIX_PATH="${torch_cmake_prefix}:${pybind11_cmake_prefix}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export TMOL_DISABLE_WHEEL_FETCH=1

"${python_bin}" -m pip install \
  --no-deps \
  --no-build-isolation \
  --config-settings=cmake.define.TMOL_ENABLE_CUDA=OFF \
  'tmol @ git+https://github.com/uw-ipd/tmol.git@v0.1.40'
