#!/usr/bin/env bash
# Added for the Aurora XPU port; see aurora/README.md.
# Install Proteina-Complexa over Aurora's XPU-enabled frameworks module.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
project_root="${AURORA_PROJECT_ROOT:-/lus/flare/projects/FRAME-IDP/${USER:-${LOGNAME:-user}}}"
env_prefix="${PROTEINA_ENV:-${project_root}/envs/proteina-complexa}"
download_weights=false

usage() {
  cat <<'EOF'
Usage: aurora/bootstrap_aurora.sh [--env VENV_PREFIX] [--weights]

Creates a venv which inherits Aurora's validated PyTorch/XPU runtime.  --weights
downloads every Proteina-Complexa and community-model checkpoint, including
AlphaFold2, ESM2, ESMFold, and RF3 (allow roughly 50 GB); omit it for a
software-only install.
EOF
}

while (($#)); do
  case "$1" in
    --env) env_prefix="$2"; shift 2 ;;
    --weights) download_weights=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

set +u
module load frameworks
set -u
source "${script_dir}/runtime_env.sh" "${env_prefix}"

if [[ ! -x "${env_prefix}/bin/python" ]]; then
  python -m venv "${env_prefix}" --system-site-packages
fi

# A normal `source VENV/bin/activate` also needs Aurora's framework module and
# project-local cache policy.  Install that behavior into the venv itself.
bash "${script_dir}/configure_venv_activation.sh" "${env_prefix}"

python_bin="${env_prefix}/bin/python"
# `--no-deps` is essential: a normal resolver sees the upstream metadata and
# tries to replace Aurora's XPU PyTorch with NVIDIA CUDA wheels.  uv is used
# only as a fast installer for this existing venv; it does not create or own
# the environment.
# `--no-config` prevents uv from reading the upstream project's cu126 package
# index.  The port must resolve auxiliary packages only from PyPI.
if ! "${python_bin}" -c '
import importlib.util
names = ("atomworks", "bioservices", "biotraj", "cachebox", "contextlib2", "cytoolz", "etils", "humanize", "hydride", "jax", "jmp", "lightning", "looseversion", "ml_dtypes", "mmtf", "modin", "narwhals", "netCDF4", "numpy", "openbabel", "openmm", "opt_einsum", "orderly_set", "orbax", "prody", "proteinfoundation", "py3Dmol", "rdkit", "rich_click", "rust_simulation_tools", "tensorstore", "toolz", "torch_geometric", "wadler_lindig", "xarray", "colabdesign")
missing = [name for name in names if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(1)

# Treat binary compatibility as part of the installation health check.  In
# particular, a later standalone ``pip install`` can otherwise upgrade NumPy
# without rebuilding the fixed SciPy 1.11.4 wheel.  That failure only appears
# when structure utilities import scipy.spatial, well after the simple module
# presence check above.
import numpy
if numpy.__version__ != "1.26.4":
    raise SystemExit(1)
import scipy.spatial
import modin.pandas
'; then
  uv pip install --no-config --index-url https://pypi.org/simple --python "${python_bin}" --no-deps --upgrade -r "${script_dir}/requirements-aurora.txt"
  uv pip install --no-config --index-url https://pypi.org/simple --python "${python_bin}" --no-deps --editable "${repo_dir}"
  uv pip install --no-config --index-url https://pypi.org/simple --python "${python_bin}" --no-deps --editable "${repo_dir}/community_models/colabdesign"
else
  echo "Aurora base profile is already installed; skipping package reinstall."
fi

# Foundry has a native XPU code path.  Its published [rf3]/[all] extras are
# nevertheless CUDA-only today, so install the base RF3 runtime plus its
# non-CUDA dependencies with pip's resolver disabled.  This is deliberate: a
# normal resolver would replace the vendor-supported XPU PyTorch with PyPI's
# NVIDIA build.
if ! "${python_bin}" -c 'import foundry, rf3'; then
  "${python_bin}" -m pip install --no-deps --upgrade -r "${script_dir}/requirements-foundry-xpu.txt"
fi

# TMol has CUDA and CPU implementations but no Intel XPU backend.  Build its
# CPU implementation so the interface-energy reward remains usable alongside
# XPU generation and evaluation.
if ! "${python_bin}" -c 'import tmol'; then
  bash "${script_dir}/install_tmol_cpu.sh" "${env_prefix}"
fi

# Complete evaluation / analysis with CPU-only tools.  Their CUDA releases
# cannot run on Aurora; this installer uses the official AVX2 CPU builds and
# project-local source/build/cache directories.
bash "${script_dir}/install_cpu_tools.sh"

# Do not use `complexa init uv`: its generated activation file assumes CUDA
# wheels and a .venv under the source tree.  runtime_env.sh supplies the Aurora
# equivalent and exports all config locations.
if [[ ! -f "${repo_dir}/.env" ]]; then
  cp "${repo_dir}/.env_example" "${repo_dir}/.env"
fi

if [[ "${download_weights}" == true ]]; then
  (
    cd "${repo_dir}"
    bash env/download_startup.sh --everything
  )
fi

echo "Installed Proteina-Complexa XPU environment: ${env_prefix}"
echo "Activate with: source ${env_prefix}/bin/activate"
