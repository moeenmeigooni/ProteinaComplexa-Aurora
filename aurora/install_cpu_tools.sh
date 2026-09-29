#!/usr/bin/env bash
# Added for the Aurora XPU port; see aurora/README.md.
# Install CPU-only external tools needed by Proteina-Complexa evaluation.
#
# Aurora has Intel GPUs, while the upstream optional GPU releases of Foldseek
# and MMseqs2 require NVIDIA CUDA.  Their official AVX2 CPU releases preserve
# all clustering/analysis functions without CUDA.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
tools_dir="${repo_dir}/tools"
bin_dir="${tools_dir}/bin"
src_dir="${tools_dir}/src"
mkdir -p "${bin_dir}" "${src_dir}"

work_dir="$(mktemp -d "${tools_dir}/.install.XXXXXX")"
cleanup() {
  rm -rf "${work_dir}"
}
trap cleanup EXIT

fetch_archive() {
  local url="$1"
  local destination="$2"
  curl --fail --location --retry 3 --retry-delay 2 --output "${destination}" "${url}"
}

if [[ ! -x "${bin_dir}/foldseek" ]]; then
  fetch_archive "https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz" "${work_dir}/foldseek.tar.gz"
  tar -xzf "${work_dir}/foldseek.tar.gz" -C "${work_dir}"
  foldseek_path="$(find "${work_dir}" -type f -path '*/bin/foldseek' -print -quit)"
  test -n "${foldseek_path}"
  install -m 0755 "${foldseek_path}" "${bin_dir}/foldseek"
fi

if [[ ! -x "${bin_dir}/mmseqs" ]]; then
  fetch_archive "https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz" "${work_dir}/mmseqs.tar.gz"
  tar -xzf "${work_dir}/mmseqs.tar.gz" -C "${work_dir}"
  mmseqs_path="$(find "${work_dir}" -type f -path '*/bin/mmseqs' -print -quit)"
  test -n "${mmseqs_path}"
  install -m 0755 "${mmseqs_path}" "${bin_dir}/mmseqs"
fi

if [[ ! -x "${bin_dir}/sc" ]]; then
  # sc-rs is the Open Source, CPU-parallel executable expected by the
  # repository's JSON CLI adapter.  Cargo state stays beneath project storage.
  if ! type module >/dev/null 2>&1; then
    echo "Aurora environment modules are required to build sc-rs." >&2
    exit 1
  fi
  set +u
  module load rust/1.86.0
  set -u
  # Aurora's rust/1.86 module presently omits libssh2 from its runtime loader
  # path even though Cargo links it.  Locate the framework-provided copy rather
  # than pinning an Aurora release-specific path.
  if ! cargo --version >/dev/null 2>&1; then
    libssh2_path="$(find /opt/aurora -type f -path '*/miniforge3-*/lib/libssh2.so.*' -print -quit)"
    if [[ -z "${libssh2_path}" ]]; then
      echo "Cargo needs libssh2, but no Aurora-provided libssh2 was found." >&2
      exit 1
    fi
    export LD_LIBRARY_PATH="$(dirname -- "${libssh2_path}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
  sc_source="${src_dir}/sc-rs"
  if [[ ! -d "${sc_source}/.git" ]]; then
    git clone --depth 1 https://github.com/cytokineking/sc-rs.git "${sc_source}"
  fi
  export CARGO_HOME="${tools_dir}/cargo-home"
  export CARGO_TARGET_DIR="${sc_source}/target"
  # Upstream sc-rs does not ship a Cargo.lock; Cargo resolves its manifest on
  # the first project-local build and leaves the generated lockfile in source.
  (cd "${sc_source}" && cargo build --release)
  install -m 0755 "${CARGO_TARGET_DIR}/release/sc" "${bin_dir}/sc"
fi

if [[ ! -x "${bin_dir}/dssp" ]]; then
  # DSSP 4.6 uses C++20 and supplies all build dependencies through CMake.
  # The version is pinned for a reproducible project-local installation.
  dssp_source="${src_dir}/dssp"
  if [[ ! -d "${dssp_source}/.git" ]]; then
    git clone --depth 1 --branch v4.6.1 https://github.com/PDB-REDO/dssp.git "${dssp_source}"
  fi
  cmake -S "${dssp_source}" -B "${dssp_source}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${tools_dir}/dssp"
  cmake --build "${dssp_source}/build" --parallel "${PROTEINA_TOOL_BUILD_JOBS:-4}"
  cmake --install "${dssp_source}/build"
  dssp_path="$(find "${tools_dir}/dssp" -type f -name mkdssp -print -quit)"
  test -n "${dssp_path}"
  install -m 0755 "${dssp_path}" "${bin_dir}/dssp"
fi

# Modern DSSP delegates residue chemistry to libcifpp.  The runtime package
# does not include the PDB Chemical Component Dictionary, so stage the
# uncompressed official dictionary in the same project-local prefix.  (The
# 4.6.1 build on Aurora does not discover the gzip form reliably.) Without
# this file DSSP starts but cannot assign standard-amino-acid structure.
dssp_data_dir="${tools_dir}/dssp/share/libcifpp"
dssp_components="${dssp_data_dir}/components.cif"
if [[ ! -s "${dssp_components}" ]]; then
  mkdir -p "${dssp_data_dir}"
  fetch_archive \
    "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz" \
    "${work_dir}/components.cif.gz"
  gzip -dc "${work_dir}/components.cif.gz" > "${work_dir}/components.cif"
  install -m 0644 "${work_dir}/components.cif" "${dssp_components}"
fi

"${bin_dir}/foldseek" version
"${bin_dir}/mmseqs" version
# sc-rs has no standalone --help mode; it reports argument usage with a
# non-zero status.  Executability is the appropriate install-time check.
test -x "${bin_dir}/sc"
"${bin_dir}/dssp" --version
