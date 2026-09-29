#!/usr/bin/env bash
# Added for the Aurora XPU port; see aurora/README.md.
# Make `source VENV/bin/activate` the complete Aurora activation command.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 VENV_PREFIX" >&2
  exit 2
fi

venv_prefix="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
activate_dir="${venv_prefix}/bin"
base_activate="${activate_dir}/activate.base"

if [[ ! -f "${base_activate}" ]]; then
  mv "${activate_dir}/activate" "${base_activate}"
fi

sed \
  -e "s|__PROTEINA_RUNTIME_ENV__|${script_dir}/runtime_env.sh|g" \
  -e "s|__PROTEINA_REPO__|${repo_dir}|g" \
  "${script_dir}/venv_activate.template" > "${activate_dir}/activate"
chmod 0644 "${activate_dir}/activate"
