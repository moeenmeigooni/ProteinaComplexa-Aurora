#!/usr/bin/env bash
# Canonical Aurora installer entrypoint. Keep bootstrap_aurora.sh for users
# who already have it in scripts or notes.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/bootstrap_aurora.sh" "$@"
