#!/usr/bin/env bash
# Added for the Aurora XPU port; see aurora/README.md.
# Source after `module load frameworks`.  It keeps all user-writable state in
# FRAME-IDP project storage rather than the real home directory.

if [[ $# -ne 1 ]]; then
  echo "Usage: source aurora/runtime_env.sh VENV_PREFIX" >&2
  return 2 2>/dev/null || exit 2
fi

proteina_env="$1"
proteina_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
proteina_env_parent="$(dirname -- "${proteina_env}")"
proteina_env_name="$(basename -- "${proteina_env}")"
proteina_cache_root="${PROTEINA_CACHE_ROOT:-${proteina_env_parent}/.${proteina_env_name}-cache}"
proteina_runtime_home="${PROTEINA_RUNTIME_HOME:-${proteina_env_parent}/.${proteina_env_name}-runtime-home}"

mkdir -p \
  "${proteina_cache_root}/pip" \
  "${proteina_cache_root}/uv" \
  "${proteina_cache_root}/xdg" \
  "${proteina_cache_root}/xdg-config" \
  "${proteina_cache_root}/xdg-data" \
  "${proteina_cache_root}/huggingface" \
  "${proteina_cache_root}/torch" \
  "${proteina_cache_root}/torch-extensions" \
  "${proteina_cache_root}/jax" \
  "${proteina_cache_root}/numba" \
  "${proteina_cache_root}/triton" \
  "${proteina_cache_root}/ray" \
  "${proteina_cache_root}/matplotlib" \
  "${proteina_cache_root}/ipython" \
  "${proteina_cache_root}/jupyter/config" \
  "${proteina_cache_root}/jupyter/data" \
  "${proteina_cache_root}/jupyter/runtime" \
  "${proteina_cache_root}/wandb/cache" \
  "${proteina_cache_root}/wandb/config" \
  "${proteina_cache_root}/wandb/data" \
  "${proteina_cache_root}/wandb/runs" \
  "${proteina_cache_root}/python-userbase" \
  "${proteina_cache_root}/tmp" \
  "${proteina_runtime_home}"

export PROTEINA_ROOT="${proteina_root}"
export PROTEINA_ENV="${proteina_env}"
export PROTEINA_ACCELERATOR=xpu
export PROTEINA_CACHE_ROOT="${proteina_cache_root}"
export PROTEINA_RUNTIME_HOME="${proteina_runtime_home}"
export HOME="${proteina_runtime_home}"
export PYTHONNOUSERSITE=1
export PYTHONUSERBASE="${proteina_cache_root}/python-userbase"
export PIP_CACHE_DIR="${proteina_cache_root}/pip"
export UV_CACHE_DIR="${proteina_cache_root}/uv"
export XDG_CACHE_HOME="${proteina_cache_root}/xdg"
export XDG_CONFIG_HOME="${proteina_cache_root}/xdg-config"
export XDG_DATA_HOME="${proteina_cache_root}/xdg-data"
export HF_HOME="${proteina_cache_root}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_ASSETS_CACHE="${HF_HOME}/assets"
export TORCH_HOME="${proteina_cache_root}/torch"
export TORCH_EXTENSIONS_DIR="${proteina_cache_root}/torch-extensions"
export JAX_COMPILATION_CACHE_DIR="${proteina_cache_root}/jax"
export NUMBA_CACHE_DIR="${proteina_cache_root}/numba"
export TRITON_CACHE_DIR="${proteina_cache_root}/triton"
export RAY_TMPDIR="${proteina_cache_root}/ray"
export MPLCONFIGDIR="${proteina_cache_root}/matplotlib"
export IPYTHONDIR="${proteina_cache_root}/ipython"
export JUPYTER_CONFIG_DIR="${proteina_cache_root}/jupyter/config"
export JUPYTER_DATA_DIR="${proteina_cache_root}/jupyter/data"
export JUPYTER_RUNTIME_DIR="${proteina_cache_root}/jupyter/runtime"
export PYTHONPYCACHEPREFIX="${proteina_cache_root}/python-pycache"
export WANDB_CACHE_DIR="${proteina_cache_root}/wandb/cache"
export WANDB_CONFIG_DIR="${proteina_cache_root}/wandb/config"
export WANDB_DATA_DIR="${proteina_cache_root}/wandb/data"
export WANDB_DIR="${proteina_cache_root}/wandb/runs"
export TMPDIR="${proteina_cache_root}/tmp"
export CACHE_DIR="${proteina_cache_root}"

# These override placeholder values in .env because python-dotenv does not
# replace an environment variable that the job has already exported.
export LOCAL_CODE_PATH="${proteina_root}"
export COMMUNITY_MODELS_PATH="${proteina_root}/community_models"
export PROTEINA_TOOLS="${proteina_root}/tools"
export PATH="${PROTEINA_TOOLS}/bin:${PATH}"
export FOLDSEEK_EXEC="${FOLDSEEK_EXEC:-${PROTEINA_TOOLS}/bin/foldseek}"
export MMSEQS_EXEC="${MMSEQS_EXEC:-${PROTEINA_TOOLS}/bin/mmseqs}"
export SC_EXEC="${SC_EXEC:-${PROTEINA_TOOLS}/bin/sc}"
export DSSP_EXEC="${DSSP_EXEC:-${PROTEINA_TOOLS}/bin/dssp}"
export AF2_DIR="${COMMUNITY_MODELS_PATH}/ckpts/AF2"
export ESM_DIR="${COMMUNITY_MODELS_PATH}/ckpts/ESM2"
export ESMFOLD_DIR="${COMMUNITY_MODELS_PATH}/ckpts/ESMFold"
export RF3_DIR="${COMMUNITY_MODELS_PATH}/ckpts/RF3"
export RF3_CKPT_PATH="${RF3_DIR}/rf3_foundry_01_24_latest_remapped.ckpt"
export RF3_EXEC_PATH="${RF3_EXEC_PATH:-${proteina_env}/bin/rf3}"
export CKPT_PATH="${proteina_root}/ckpts"
export DATA_PATH="${DATA_PATH:-${proteina_root}/assets}"
