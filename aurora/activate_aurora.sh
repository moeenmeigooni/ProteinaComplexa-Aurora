#!/usr/bin/env bash
# Added for the Aurora XPU port; see aurora/README.md.
# Compatibility wrapper. Prefer `source .../envs/proteina-complexa/bin/activate`.

set -euo pipefail
project_root="${AURORA_PROJECT_ROOT:-/lus/flare/projects/FRAME-IDP/${USER:-${LOGNAME:-user}}}"
env_prefix="${PROTEINA_ENV:-${project_root}/envs/proteina-complexa}"
source "${env_prefix}/bin/activate"
