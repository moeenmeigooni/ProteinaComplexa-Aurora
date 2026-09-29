# Proteina-Complexa on ALCF Aurora

This port targets Aurora's Intel Data Center GPU Max tiles.  It does not use
CUDA, NVIDIA containers, CUDA PyTorch wheels, or CUDA-only PyG extensions.

## Installation

Clone the Aurora port and enter its checkout:

```bash
git clone https://github.com/moeenmeigooni/ProteinaComplexa-Aurora.git
cd ProteinaComplexa-Aurora
```

Set `AURORA_PROJECT_ROOT` to a shared project directory visible from both the
login and compute nodes. By default, the scripts use
`/lus/flare/projects/FRAME-IDP/$USER`. The checkout can live anywhere on shared
storage; the environment defaults to
`${AURORA_PROJECT_ROOT}/envs/proteina-complexa`.

On an Aurora UAN, run:

```bash
export AURORA_PROJECT_ROOT="/lus/flare/projects/FRAME-IDP/${USER}"
bash aurora/install_aurora.sh
```

This creates a `venv --system-site-packages` over `module load frameworks`.
That module owns the site-supported PyTorch/XPU and Level Zero runtime, so the
bootstrap deliberately uses `pip --no-deps` and never installs `torch`,
`triton`, or an NVIDIA CUDA package into the venv.

To stage every Proteina-Complexa and community-model checkpoint (protein,
ligand, and AME Complexa; ProteinMPNN; LigandMPNN; AF2; ESM2; ESMFold; and
RF3), add `--weights`.  Allow roughly 50 GB. The downloads remain under the
source checkout and are validated before being accepted as cached.

```bash
bash aurora/install_aurora.sh --weights
```

## Activation and cache policy

Activate the environment in a fresh shell or PBS job with one command:

```bash
source aurora/activate_aurora.sh
```

The standard venv activation loads Aurora's supported framework module and
redirects `HOME`, Pip/UV/XDG, Hugging Face,
Transformers, Torch, JAX, Matplotlib, Weights & Biases, temporary files, and
Python user-site state under `${AURORA_PROJECT_ROOT}/envs/`.
It also exports the correct project paths for model weights and the installed
Foldseek, MMseqs2, DSSP, and `sc-rs` executables.  Override
`PROTEINA_CACHE_ROOT` or `PROTEINA_RUNTIME_HOME` only with another
project-visible location.  It is safe in PBS scripts that set `set -u`.

## Execution and validation

Run the XPU smoke test before a design campaign:

```bash
cd /path/to/ProteinaComplexa-Aurora
qsub aurora/smoke_xpu.pbs
```

It runs a native PyTorch XPU kernel, the replacement for the CUDA-only
`torch_scatter` reduction, JAX via Intel Extension for OpenXLA (`sycl`),
OpenMM's OpenCL discovery, and a Lightning prediction through the local XPU
accelerator shim.

After all checkpoints are staged, validate the optional model backends with:

```bash
qsub aurora/full_models_xpu.pbs
```

That job performs one LigandMPNN design and one low-step RF3 fold on an XPU
tile.  It is an installation test, not a scientifically meaningful prediction.
Run `generate_smoke_xpu.pbs` successfully first: this job also evaluates the
resulting real two-chain design with TMol's CPU-only interface-energy backend.
If the smoke target PDB has missing side-chain atoms, set `TMOL_TEST_PDB` to a
complete two-chain complex before submitting this job; otherwise it scores the
generated structure automatically.
TMol has no XPU kernel, so that one component intentionally runs on CPU while
the generative and folding models remain on XPU.

Validate the CPU evaluation layer separately with:

```bash
qsub aurora/evaluation_tools_smoke.pbs
```

It runs Proteina's public interface-metric API (real SC plus SASA), DSSP
secondary-structure calculation, and both Foldseek and MMseqs2 diversity
paths against a bundled two-chain structure.

To exercise both checkpointed non-protein variants, submit:

```bash
qsub aurora/variant_generation_xpu.pbs
```

This runs one low-step Ligand Complexa sample and one low-step AME
(motif-plus-ligand) sample on an XPU tile.

Finally, verify the default protein-binder AF2 reward itself (rather than only
JAX device discovery) with:

```bash
qsub aurora/af2_reward_xpu.pbs
```

It runs a real AF2-Multimer reward over a bundled two-chain complex through
Intel Extension for OpenXLA's `sycl` backend and writes its refolded PDB under
`aurora/af2-reward-xpu-smoke/`.

For a protein-binder design, submit one process per tile and start with the
upstream serial configuration:

```bash
export ZE_AFFINITY_MASK=0
complexa design configs/search_binder_local_pipeline.yaml \
  ++gen_njobs=1 ++eval_njobs=1
```

The pipeline needs a real target structure and `DATA_PATH` appropriate to that
target.  Its default AF2 reward uses JAX's `sycl` backend.  If a PBS script
launches multiple independent processes, bind each with a different
`ZE_AFFINITY_MASK`; the CLI also maps its `gen_njobs` children to tile indices.

## Ported paths and limitations

| Component | Aurora implementation |
| --- | --- |
| Main Proteina model, ProteinMPNN, ESM2, and ESMFold | Native PyTorch `torch.xpu` |
| Lightning inference | Local single-device XPU accelerator shim |
| AF2 reward and sequence hallucination | JAX 0.4.38 + Intel Extension for OpenXLA 0.6.0 (`sycl`) using `oneapi/release/2025.3.1`. Aurora PE 26.181.0's default oneAPI 2026.1 runtime does not provide the older oneMKL/SYCL sonames required by this plugin. Activation keeps the framework's 2026 shared-library paths alongside the compatible 2025 paths so PyTorch 2.13 and AF2 JAX can load in the same environment. Aurora's backend lacks the SVD primitive used by ColabDesign's in-graph Kabsch superposition, so XPU scoring skips that non-reward operation to preserve gradients. When a SYCL binder refold is saved, `rust-simulation-tools` performs the same target C-alpha Kabsch superposition as CPU post-processing. Native AF2 inference, pLDDT, PAE, dgram CCE, FAPE, gradients, and aligned PDB output remain enabled. |
| PyG use | Base `torch_geometric`; CUDA-only `torch_scatter` call replaced with native Torch |
| OpenMM relaxation | OpenCL platform on Intel GPUs, with CPU fallback if OpenCL is absent |
| RF3 / `rc-foundry` | Foundry 0.2 XPU mode with native Torch fallback. Its CUDA-only cuEquivariance extras are intentionally not installed. |
| LigandMPNN's ProDy-based ligand preparation | Native `torch.xpu` with ProDy 2.6.1 built for the Python 3.12 environment |
| TMol interface-force-field reward | CPU-only build and automatic CPU fallback. It supports scoring and gradients; TMol itself has no XPU kernel, so it is slower and intentionally offloads this component to CPU. |
| Foldseek / MMseqs2 diversity and novelty | Official AVX2 CPU builds. Their CUDA releases are intentionally not used. |
| Shape complementarity | Project-built CPU-parallel `sc-rs` executable. |
| DSSP secondary structure | Project-built DSSP 4.6.1 CPU executable. |

The upstream CUDA UV builder and NVIDIA Dockerfile remain unsuitable for
Aurora.  Keep the vendor-supplied framework module loaded whenever this venv
is active.
