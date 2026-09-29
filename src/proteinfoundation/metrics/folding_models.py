# Modified for Aurora XPU compatibility by the FRAME-IDP Aurora port; see aurora/README.md.
import os
import shutil
import subprocess
from typing import Literal

import torch
from loguru import logger
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers import logging as hf_logging
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

from proteinfoundation.utils.device_utils import torch_device

hf_logging.set_verbosity_error()


def create_individual_fasta_files(
    sequences: list[str],
    output_dir: str,
    format_type: Literal["simple"] = "simple",
    name_prefix: str = "seq",
) -> str:
    """
    Creates individual FASTA files for each sequence with appropriate headers for different folding models.

    Args:
        sequences: List of protein sequences
        output_dir: Directory where individual FASTA files will be written
        format_type: Header format to use:
            - "simple": >seq_1, >seq_2, ... (for ESMFold/ColabFold)
        name_prefix: Prefix for sequence names

    Returns:
        Path to the directory containing the individual FASTA files
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, seq in enumerate(sequences):
        seq_name = f"{name_prefix}_{i + 1}"
        fasta_path = os.path.join(output_dir, f"{seq_name}.fasta")

        if format_type == "simple":
            header = f">{seq_name}"
        else:
            raise ValueError(f"Unknown format_type: {format_type}")

        with open(fasta_path, "w") as f:
            f.write(f"{header}\n{seq}\n")

    return output_dir


def run_esmfold(
    sequences: list[str],
    path_to_esmfold_out: str,
    name: str,
    suffix: str,
    cache_dir: str | None = None,
    keep_outputs: bool = False,
) -> list[str]:
    """
    Runs ESMFold on sequences and stores results as PDB files.

    For now, runs with a single GPU, though not a big deal if we parallelie jobs (easily
    done with our inference pipeline).

    Args:
        sequences: List of protein sequences to predict
        path_to_esmfold_out: Root directory to store outputs of ESMFold as PDBs
        name: name to use when storing
        suffix: to use as suffix when storing files
        cache_dir: Cache directory for model weights
        keep_outputs: Whether to keep individual output directories after processing.
            If False (default), temporary directories are deleted to save space.

    Returns:
        List of paths (list of str) to PDB files
    """
    # PBS jobs on Aurora do not set SLURM_JOB_ID.  Always honor the explicit
    # project-local ESMFold cache so model downloads and runtime loads never
    # fall back to the user's home directory.
    final_cache_dir = cache_dir or os.environ.get("ESMFOLD_DIR") or os.environ.get("CACHE_DIR")
    if final_cache_dir:
        final_cache_dir = os.path.expanduser(final_cache_dir)

    # Compute nodes are intentionally offline.  The installer stages this exact
    # revision into ESMFOLD_DIR, so never allow a lazy network fetch (or a
    # fallback cache under HOME) during an inference job.
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esmfold_v1", cache_dir=final_cache_dir, local_files_only=True
    )
    esm_model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", cache_dir=final_cache_dir, local_files_only=True
    )
    device = torch_device(require=True)
    esm_model = esm_model.to(device)

    # Run ESMFold
    list_of_strings_pdb = []
    if len(sequences) == 8:
        max_nres = max([len(x) for x in sequences])
        if max_nres > 700:
            batch_size = 1
            num_batches = 8
        elif max_nres > 500:
            batch_size = 2
            num_batches = 4
        elif max_nres > 200:
            batch_size = 4
            num_batches = 2
        else:
            batch_size = 8
            num_batches = 1
    elif len(sequences) == 1:
        batch_size = 8
        num_batches = 1
    else:
        raise OSError("We can only run ESMFold with 1 or 8 sequences... We should fix this...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        inputs = tokenizer(
            sequences[start_idx:end_idx],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        inputs = {k: value.to(device) for k, value in inputs.items()}

        with torch.no_grad():
            _outputs = esm_model(**inputs)

        _list_of_strings_pdb = _convert_esm_outputs_to_pdb(_outputs)
        list_of_strings_pdb.extend(_list_of_strings_pdb)

    # Create out directory if not there
    if not os.path.exists(path_to_esmfold_out):
        os.makedirs(path_to_esmfold_out)

    # Store generations for each sequence
    out_esm_paths = []
    for i, pdb in enumerate(list_of_strings_pdb):
        fname = f"esm_{i + 1}.pdb_esm_{suffix}"
        fdir = os.path.join(path_to_esmfold_out, fname)
        with open(fdir, "w") as f:
            f.write(pdb)
            out_esm_paths.append(fdir)

    # Unlike the ColabFold runner below, ESMFold does not create an
    # ``individual_fastas`` temporary directory.  Never derive a cleanup
    # target from the requested output path: with an absolute output directory
    # the old two-parent calculation could target the caller's parent folder.
    # ``keep_outputs`` is retained for API compatibility; the PDB outputs are
    # always preserved.
    if not keep_outputs:
        logger.debug("ESMFold creates no temporary files to clean up")

    return out_esm_paths


# I got this function from hugging face's ESM notebook example
def _convert_esm_outputs_to_pdb(outputs) -> list[str]:
    """Takes ESMFold outputs and converts them to a list of PDBs (as strings)."""
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def run_colabfold(
    sequences: list[str],
    path_to_colabfold_out: str,
    suffix: str = "",
    relax: bool = False,
    cache_dir: str | None = None,
    keep_outputs: bool = False,
) -> list[str]:
    """
    Runs ColabFold batch on sequences using individual FASTA files and returns paths to top-ranked PDB files.

    Args:
        sequences (List[str]): List of protein sequences to predict
        path_to_colabfold_out (str): Output directory path for ColabFold results.
        suffix (str): Suffix to add to output files to indicate source (e.g. "mpnn" or "pdb")
        relax (bool): whether to relax the structure afterwards
        cache_dir (Optional[str]): Cache directory for model weights
        keep_outputs (bool): Whether to keep individual output directories after processing.
            If False (default), temporary directories are deleted to save space.

    Returns:
        list[str]: Paths to top-ranked PDB files by pLDDT (rank_001) in the order of input sequences.

    Raises:
        RuntimeError: If ColabFold command fails.
    """
    # Create output directory if it doesn't exist
    os.makedirs(path_to_colabfold_out, exist_ok=True)
    os.makedirs(os.path.join(path_to_colabfold_out, "structures"), exist_ok=True)

    cache_dir = os.environ.get("CACHE_DIR")
    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)
        os.environ["XDG_CACHE_HOME"] = cache_dir

    # Create individual FASTA files using the unified function
    fasta_dir = os.path.join(path_to_colabfold_out, "individual_fastas")
    create_individual_fasta_files(sequences, fasta_dir, format_type="simple")

    # Get sequence names for output parsing
    seq_names = [f"seq_{i + 1}" for i in range(len(sequences))]

    # Run ColabFold batch on the directory containing individual FASTA files
    batch_command = (
        f"colabfold_batch {fasta_dir} {path_to_colabfold_out}/structures --msa-mode single_sequence --data {cache_dir}"
    )
    if relax:
        batch_command = batch_command + " --num-relax 1 --use-gpu-relax"

    try:
        result = subprocess.run(batch_command, shell=True, check=True)
        if result.returncode != 0:
            logger.error(f"ColabFold command failed with error: {result.stderr}")
            raise RuntimeError(f"ColabFold command failed: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ColabFold command failed with error: {e.stderr}")
        raise RuntimeError(f"ColabFold command failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error running ColabFold: {e!s}")
        raise RuntimeError(f"Unexpected error running ColabFold: {e!s}")

    # Collect PDB file paths for rank_001 models in the original sequence order
    pdb_file_paths = []
    for seq_name in seq_names:
        found_pdb = False
        for filename in os.listdir(f"{path_to_colabfold_out}/structures"):
            if filename.startswith(seq_name) and "rank_001" in filename and filename.endswith(".pdb"):
                pdb_path = f"{path_to_colabfold_out}/structures/{filename}"
                if suffix:
                    # Add suffix to the filename
                    new_path = pdb_path.replace(".pdb", f"_{suffix}.pdb")
                    shutil.copy(pdb_path, new_path)
                    pdb_file_paths.append(new_path)
                else:
                    pdb_file_paths.append(pdb_path)
                found_pdb = True
                break

        if not found_pdb:
            logger.warning(f"No rank_001 PDB file found for sequence: {seq_name}")
            pdb_file_paths.append(None)

    # Clean up individual FASTA files directory
    if not keep_outputs:
        try:
            shutil.rmtree(fasta_dir)
        except Exception as e:
            logger.warning(f"Could not remove FASTA directory: {e}")

    return pdb_file_paths
