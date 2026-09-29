# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
"""AF2 Reward Model for protein design optimization.

This module provides a reward model based on AlphaFold2 for evaluating
and optimizing protein sequences and structures.
"""

import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import jax
import jax.dlpack
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.dlpack
from colabdesign import mk_afdesign_model
from omegaconf import DictConfig

from proteinfoundation.rewards.alphafold2_reward_utils import (
    add_alignment_bb_ca_loss,
    add_helix_binder_loss,
    add_i_ptm_energy_loss,
    add_i_ptm_loss,
    add_rg_loss,
    add_termini_distance_loss,
)
from proteinfoundation.rewards.base_reward import REWARD_KEY, TOTAL_REWARD_KEY, BaseRewardModel, standardize_reward
from proteinfoundation.utils.device_utils import current_device_index, get_jax_device
from proteinfoundation.utils.pdb_utils import from_pdb_file


def _load_pdb_atom_models(
    path: str,
) -> tuple[list[str], list[list[tuple[int, str, str, np.ndarray]]]]:
    """Read coordinate-bearing protein records, preserving PDB text layout."""
    lines = Path(path).read_text().splitlines(keepends=True)
    models: list[list[tuple[int, str, str, np.ndarray]]] = []
    current: list[tuple[int, str, str, np.ndarray]] = []
    for index, line in enumerate(lines):
        if line.startswith("ATOM  "):
            current.append(
                (
                    index,
                    line[21],
                    line[12:16].strip(),
                    np.asarray(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64
                    ),
                )
            )
        elif line.startswith("ENDMDL") and current:
            models.append(current)
            current = []
    if current:
        models.append(current)
    if not models:
        raise ValueError(f"No ATOM records found in {path}")
    return lines, models


def _realign_sycl_saved_pdb(output_pdb: str, reference_pdb: str, target_chain: str) -> tuple[float, float]:
    """Restore an AF2 PDB's target coordinate frame with Rust Kabsch alignment.

    Intel OpenXLA cannot compile the SVD used by ColabDesign's in-graph
    Kabsch path.  This runs only after XPU inference has finished, so it never
    moves a differentiable reward calculation off SYCL.  The Rust routine
    aligns each written model on target C-alpha atoms and transforms every
    output atom into the input target's coordinate frame.
    """
    try:
        from rust_simulation_tools import kabsch_align
    except ImportError as exc:  # pragma: no cover - bootstrap installs this dependency
        raise RuntimeError(
            "rust-simulation-tools is required to realign AF2 SYCL PDB output. "
            "Run aurora/bootstrap_aurora.sh to install the Aurora profile."
        ) from exc

    target_chains = tuple(chain.strip() for chain in target_chain.split(",") if chain.strip())
    if not target_chains:
        raise ValueError("A non-empty target_chain is required for SYCL PDB realignment")

    output_lines, output_models = _load_pdb_atom_models(output_pdb)
    _, reference_models = _load_pdb_atom_models(reference_pdb)
    reference_atoms = reference_models[0]
    target_set = set(target_chains)
    ordered_target_atoms = [atom for chain in target_chains for atom in reference_atoms if atom[1] == chain]
    if not ordered_target_atoms:
        raise ValueError(f"No protein atoms found for target chain(s) {target_chains!r} in {reference_pdb}")
    ordered_reference_atoms = ordered_target_atoms + [atom for atom in reference_atoms if atom[1] not in target_set]

    n_atoms = len(output_models[0])
    if any(len(model) != n_atoms for model in output_models):
        raise ValueError("AF2 output PDB models have inconsistent atom counts")
    trajectory = np.ascontiguousarray(
        np.asarray([[atom[3] for atom in model] for model in output_models], dtype=np.float64)
    )

    # Usual binder-refolding case: ColabDesign writes target atoms first,
    # followed by binder atoms, matching this reordered PDB reference.  If a
    # caller supplies only the target PDB plus a new sequence, construct a
    # full-length reference from the first prediction; the Rust routine uses
    # only the selected target C-alphas to calculate its transform.
    if len(ordered_reference_atoms) == n_atoms:
        reference = np.asarray([atom[3] for atom in ordered_reference_atoms], dtype=np.float64)
        align_indices = np.asarray(
            [i for i, atom in enumerate(ordered_reference_atoms) if atom[1] in target_set and atom[2] == "CA"],
            dtype=np.int64,
        )
    elif len(ordered_target_atoms) <= n_atoms:
        reference = trajectory[0].copy()
        reference[: len(ordered_target_atoms)] = np.asarray([atom[3] for atom in ordered_target_atoms], dtype=np.float64)
        align_indices = np.asarray(
            [i for i, atom in enumerate(ordered_target_atoms) if atom[2] == "CA"], dtype=np.int64
        )
    else:
        raise ValueError(
            f"AF2 output has {n_atoms} atoms, but the target reference has {len(ordered_target_atoms)} "
            f"and full reference has {len(ordered_reference_atoms)} atoms."
        )
    if len(align_indices) < 3:
        raise ValueError("At least three target C-alpha atoms are required for Kabsch realignment")

    target_before = trajectory[:, align_indices]
    aligned = kabsch_align(trajectory, np.ascontiguousarray(reference), np.ascontiguousarray(align_indices))[0 : len(output_models)]
    target_after = aligned[:, align_indices]
    target_reference = reference[align_indices]
    rmsd_before = float(np.sqrt(np.square(target_before - target_reference).sum(axis=-1).mean()))
    rmsd_after = float(np.sqrt(np.square(target_after - target_reference).sum(axis=-1).mean()))

    for model, coordinates in zip(output_models, aligned, strict=True):
        for (line_index, _, _, _), (x, y, z) in zip(model, coordinates, strict=True):
            line = output_lines[line_index]
            output_lines[line_index] = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
    Path(output_pdb).write_text("".join(output_lines))
    logger.info("Rust Kabsch realigned SYCL AF2 PDB target C-alpha RMSD %.3f -> %.3f Å", rmsd_before, rmsd_after)
    return rmsd_before, rmsd_after


class AF2RewardModel(BaseRewardModel):
    """AlphaFold2-based reward model for protein sequence optimization.

    This class implements a reward model that uses AlphaFold2 to evaluate
    protein sequences and provide gradients for optimization. It supports
    various reward components including pLDDT, PAE, and structural metrics.
    """

    IS_FOLDING_MODEL = True
    SUPPORTS_GRAD = True
    SUPPORTS_SAVE_PDB = True

    reward_options = {
        "binder": (
            "plddt",
            "pae",
            "exp_res",
            "con",
            "i_con",
            "i_pae",
            "rg",
            "i_ptm",
            "i_ptm_energy",
            "nc_termini",
            "helix_binder",
            "alignment_bb_ca_binder",
            "dgram_cce",
            "min_ipae",
            "min_ipsae",
            "avg_ipsae",
            "max_ipsae",
            "min_ipsae_10",
            "max_ipsae_10",
            "avg_ipsae_10",
        ),
        "hallucination": ("plddt", "pae", "exp_res", "con", "helix", "alignment_bb_ca"),
    }

    def __init__(
        self,
        protocol: str,
        af_params_dir: str,
        reward_weights: dict[str, float],
        use_multimer: bool = False,
        model_nums: list[int] | None = None,
        num_recycles: int = 3,
        use_initial_guess: bool = False,
        use_initial_atom_pos: bool = False,
        seed: int = 0,
        device_id: int | None = None,
    ) -> None:
        """Initialize the AF2RewardModel.

        Args:
            protocol: Protocol to use for the model.
            af_params_dir: Directory containing AlphaFold2 parameters.
            reward_weights: Dictionary of weights for different loss components.
            use_multimer: Whether to use multimer model.
            model_nums: List of model numbers to use.
            num_recycles: Number of recycles for validation.
            use_initial_guess: Whether to use initial guess.
            use_initial_atom_pos: Whether to use initial atom positions.
            seed: Random seed for reproducibility.
            device_id: GPU device ID to use. If None, auto-detects the current CUDA or XPU device.
        """
        if device_id is None:
            device_id = current_device_index()

        self.protocol = protocol
        assert protocol in [
            "hallucination",
            "binder",
        ], f"Protocol must be either 'hallucination' or 'binder', but got {protocol}"
        self.num_recycles = num_recycles
        self.af_params_dir = af_params_dir
        self.use_multimer = use_multimer
        self.model_nums = model_nums if model_nums is not None else [0, 1, 2, 3, 4]
        # Convert DictConfig to dict if necessary
        if isinstance(reward_weights, DictConfig):
            self.reward_weights = dict(reward_weights)
        else:
            self.reward_weights = reward_weights
        self.use_initial_guess = use_initial_guess
        self.use_initial_atom_pos = use_initial_atom_pos
        for reward_name in reward_weights:
            assert reward_name in self.reward_options[protocol], (
                f"Invalid reward name: {reward_name} for protocol {protocol}"
            )
        for reward_name in self.reward_options[protocol]:
            if reward_name not in self.reward_weights:
                self.reward_weights[reward_name] = 0.0
        self.device = get_jax_device(jax, device_id)
        self.seed = seed
        self.rng = random.Random(seed)

        # Initialize the AF2 model
        self.model = mk_afdesign_model(
            protocol=protocol,
            use_multimer=use_multimer,
            num_recycles=num_recycles,
            data_dir=af_params_dir,
            use_initial_guess=use_initial_guess,
            use_initial_atom_pos=use_initial_atom_pos,
            learning_rate=1.0,  # No learning rate is used in the reward model
            device=self.device,
        )

        # Intel Extension for OpenXLA's SYCL backend does not currently lower
        # the ``eigh`` primitive used by ColabDesign's SVD-based Kabsch
        # realignment.  That realignment is cosmetic for this reward: pLDDT,
        # PAE and all design losses are computed before it. Disable only the
        # in-graph alignment on Aurora instead of moving AF2 inference to CPU
        # or falling back to CUDA; saved PDBs are realigned afterward by the
        # Rust/NumPy helper above, outside the differentiable JAX graph.
        if getattr(self.device, "platform", None) == "sycl":
            self.model._args["realign"] = False
            self.model._args["skip_kabsch"] = True

        if reward_weights.get("alignment_bb_ca", 0) != 0:
            add_alignment_bb_ca_loss(self.model, reward_weights["alignment_bb_ca"])

        if reward_weights.get("alignment_bb_ca_binder", 0) != 0:
            add_alignment_bb_ca_loss(self.model, reward_weights["alignment_bb_ca_binder"], binder_only=True)

        if reward_weights.get("helix_binder", 0) != 0:
            add_helix_binder_loss(self.model, reward_weights["helix_binder"])

        if reward_weights.get("rg", 0) != 0:
            add_rg_loss(self.model, reward_weights["rg"])

        if reward_weights.get("nc_termini", 0) != 0:
            add_termini_distance_loss(self.model, reward_weights["nc_termini"])

        if reward_weights.get("i_ptm", 0) != 0:
            add_i_ptm_loss(self.model, reward_weights["i_ptm"])

        if reward_weights.get("i_ptm_energy", 0) != 0:
            add_i_ptm_energy_loss(self.model, reward_weights["i_ptm_energy"])

    def score(
        self,
        pdb_path: str,
        requires_grad: bool = False,
        sequence: torch.Tensor | None = None,
        structure: torch.Tensor | None = None,
        binder_chain: str | None = None,
        target_chain: str | None = None,
        save_pdb: bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Calculate reward and gradients for a given sequence and structure.

        Args:
            pdb_path: Path to PDB file (required)
            requires_grad: Whether to calculate gradients
            sequence: Optional input sequence tensor (logits or one-hot encoded), shape (L, 20).
                If protocol is "hallucination", this is the whole sequence.
                If protocol is "binder", this is the binder sequence.
                If None and protocol is "binder", will extract from PDB file.
            structure: Optional generated coordinates, shape (L, 37, 3). Given in ang.
                If protocol is "binder", this is the binder structure.
                If None and protocol is "binder", will extract from PDB file.
            binder_chain: Optional binder chain ID (required for binder protocol when extracting from PDB)
            target_chain: Optional target chain ID (required for binder protocol)
            save_pdb: Optional flag to save refolded PDB file (supported for AF2)

        Returns:
            Dictionary with reward components and gradients:
                reward: Dict[str, float]  # Reward values for each reward function
                grad: Dict[str, torch.Tensor]  # Gradients for sequence and structure
                plddt: torch.Tensor, shape (L,)  # pLDDT values
                pae: torch.Tensor, shape (L, L)  # PAE values
                ptm: torch.Tensor, shape (1,)  # pTM values
                total_reward: torch.Tensor, shape (1,)  # total reward

        Raises:
            Exception: Re-raises any exception after cleaning up JAX state.
        """
        # Handle binder protocol: extract from PDB if sequence/structure not provided
        if self.protocol == "binder":
            assert target_chain is not None, "target_chain is required for binder protocol"

            # If sequence is not provided, extract from PDB
            if sequence is None:
                assert binder_chain is not None, "binder_chain is required when extracting sequence from PDB"
                assert not requires_grad, "Binder sequence logits are required for gradient calculation"

                binder = from_pdb_file(pdb_path, chain_id=binder_chain)
                seq = F.one_hot(torch.tensor(binder.aatype), num_classes=20).float()
                seq = seq * 1e9

                # Extract structure if not provided
                if structure is None:
                    struct = binder.atom_positions
                else:
                    struct = structure.detach().cpu().numpy()
            else:
                seq = sequence
                struct = structure.detach().cpu().numpy() if structure is not None else None
        else:
            # Hallucination protocol
            assert sequence is not None, "sequence is required for hallucination protocol"
            seq = sequence
            struct = structure.detach().cpu().numpy() if structure is not None else None

        try:
            # Prepare model inputs
            if self.protocol == "hallucination":
                self.model.prep_inputs(
                    length=len(seq),
                    seq=seq,  #  need to pass it here, see `_prep_binder` in `af/prep.py`
                    struct=struct,
                    seed=self.rng.randint(0, 2**32 - 1),
                )
            else:
                # `prep_inputs` will call `_prep_binder` in `af/prep.py`
                # `_prep_binder` will call `_prep_model` in `af/prep.py`
                # `_prep_model` will call `restart` in `af/design.py`
                # `restart` will set opt, seq, and weights
                prep_kwargs = {
                    "pdb_filename": pdb_path,
                    "target_chain": target_chain,
                    "binder_len": len(seq),
                    "seq": seq,  #  need to pass it here, see `_prep_binder` in `af/prep.py`
                    "struct": struct,
                    "seed": self.rng.randint(0, 2**32 - 1),
                    "rm_target": False,
                    "rm_target_seq": False,
                    "rm_target_sc": False,  # remove target coordinates, sequence, and sidechain info from template, hardcoded here
                    "hotspot": None,  # not supported now
                }

                # If extracting from PDB, use binder template
                if binder_chain is not None and sequence is None:
                    prep_kwargs["binder_chain"] = binder_chain
                    prep_kwargs["use_binder_template"] = self.use_initial_atom_pos or self.use_initial_guess
                    prep_kwargs["rm_template_ic"] = self.use_initial_atom_pos or self.use_initial_guess
                else:
                    # When sequence is provided, don't use binder template
                    prep_kwargs["use_binder_template"] = False
                    prep_kwargs["rm_template_ic"] = False

                self.model.prep_inputs(**prep_kwargs)

            self.model.set_opt(
                hard=False,
                soft=True,
                temp=1.0,
                dropout=False,
                pssm_hard=False,
                weights=self.reward_weights,
            )  # transform logits to probs inide, see `soft_seq` in `shared/model.py`

            # Run prediction or optimization
            # sample_models - whether to randomly choose a model for prediction
            # we enable to randomly choose a model for prediction when requires_grad is True, i.e., in the training mode.
            sample_models = requires_grad if sequence is not None else False
            self.model.run(
                num_recycles=self.num_recycles,
                sample_models=sample_models,
                models=self.model_nums,
                backprop=requires_grad,
            )

            if save_pdb:
                save_pdb_filename = kwargs.get("output_pdb_path", pdb_path.replace(".pdb", "_refolded.pdb"))
                self.model.save_pdb(save_pdb_filename)
                if (
                    getattr(self.device, "platform", None) == "sycl"
                    and self.protocol == "binder"
                    and target_chain is not None
                ):
                    _realign_sycl_saved_pdb(save_pdb_filename, pdb_path, target_chain)

            # Extract results
            reward_dict = self.extract_results(self.model.aux)

            # Log scalar results
            logger.info("AF2RewardModel score results:")
            for k, v in reward_dict[REWARD_KEY].items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    logger.info("  %s: %.4f", k, v.item())
            logger.info("total_reward: %.4f", reward_dict[TOTAL_REWARD_KEY].item())
            logger.info("--------------------------------")
            return reward_dict

        except Exception:
            self._cleanup_jax_state()
            raise
        finally:
            # Clear model state
            self._clear_model_state()

    def extract_results(self, aux: dict[str, Any]) -> dict[str, Any]:
        """Extract reward dictionary and scores from model auxiliary output.

        Args:
            aux: Auxiliary output from the AF2 model.

        Returns:
            Dictionary with reward components and gradients
        """
        reward_components = {}
        for key in aux["losses"]:
            reward_components[key] = torch.tensor(aux["losses"][key], dtype=torch.float32)

        total_reward = torch.tensor(0.0, dtype=torch.float32)
        for key, weight in self.reward_weights.items():
            total_reward += reward_components.get(key, 0.0) * weight

        # Append raw confidence scores from aux["log"] (always populated by
        # ColabDesign) with a _log suffix so they are saved alongside
        # losses but clearly not part of the reward computation.
        # There is a difference between the loss/reward and the actual score from the log
        log = aux.get("log", {})
        for metric_key in log.keys():
            log_value = log[metric_key]
            if isinstance(log_value, torch.Tensor):
                log_value = log_value.item()
            elif isinstance(log_value, list):
                log_value = log_value[0]
            elif isinstance(log_value, np.ndarray):
                log_value = log_value.item() if log_value.ndim == 0 else float(log_value.mean())
            else:
                log_value = float(log_value)
            reward_components[f"{metric_key}_log"] = torch.tensor(log_value, dtype=torch.float32)
        grad_dict = {}
        if "grad" in aux:
            jax_grad_seq = aux["grad"].get("seq", None)
            if jax_grad_seq is not None:
                if isinstance(jax_grad_seq, jax.Array):
                    grad_dict["sequence"] = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_grad_seq))
                else:
                    grad_dict["sequence"] = torch.from_numpy(jax_grad_seq)

            jax_grad_struct = aux["grad"].get("struct", None)
            if jax_grad_struct is not None:
                if isinstance(jax_grad_struct, jax.Array):
                    grad_dict["structure"] = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_grad_struct))
                else:
                    grad_dict["structure"] = torch.from_numpy(jax_grad_struct)

        return standardize_reward(
            reward=reward_components,
            grad=grad_dict,
            total_reward=total_reward,
            plddt=torch.from_numpy(aux["plddt"]),
            pae=torch.from_numpy(aux["pae"]),
            ptm=torch.tensor(aux["ptm"], dtype=torch.float32),
        )

    def _clear_model_state(self) -> None:
        """Clear internal model state dictionaries."""
        if hasattr(self, "model"):
            if hasattr(self.model, "_inputs"):
                self.model._inputs.clear()
            if hasattr(self.model, "aux"):
                self.model.aux.clear()
            if hasattr(self.model, "_tmp"):
                self.model._tmp.clear()

    def _cleanup_jax_state(self) -> None:
        """Clean up JAX caches and backends."""
        jax.clear_caches()
        # ``jax.clear_backends`` was removed in JAX 0.4.36.  Clearing compiled
        # caches is the supported cleanup API and works for both CUDA and SYCL.

    def cleanup(self) -> None:
        """Explicit cleanup of JAX and model memory.

        Call this method to free up GPU memory and clear JAX state.
        """
        self._clear_model_state()
        # self._cleanup_jax_state()
        # gc.collect()
