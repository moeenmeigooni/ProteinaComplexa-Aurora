# Added for the Aurora XPU port; see aurora/README.md.
"""Portable accelerator selection for Proteina-Complexa.

The upstream release assumes an NVIDIA CUDA device everywhere.  Aurora uses
Intel Data Center GPU Max devices exposed by upstream PyTorch as ``xpu``.
Centralising the selection keeps CUDA installations unchanged and makes the
XPU path explicit rather than silently falling back to CPU.
"""

from __future__ import annotations

from typing import Any

import torch


def xpu_is_available() -> bool:
    """Return whether PyTorch can access an Intel XPU."""
    return bool(hasattr(torch, "xpu") and torch.xpu.is_available())


def accelerator_name() -> str:
    """Return the available accelerator in CUDA, XPU, CPU priority order."""
    if torch.cuda.is_available():
        return "cuda"
    if xpu_is_available():
        return "xpu"
    return "cpu"


def require_accelerator() -> str:
    """Return the active accelerator, rejecting accidental CPU execution."""
    accelerator = accelerator_name()
    if accelerator == "cpu":
        raise RuntimeError(
            "No supported accelerator is visible. Run on a CUDA GPU or on Aurora "
            "inside a GPU-tile job with module load frameworks."
        )
    return accelerator


def torch_device(device_id: int = 0, require: bool = False) -> torch.device:
    """Return the selected PyTorch device."""
    accelerator = require_accelerator() if require else accelerator_name()
    return torch.device(accelerator, device_id) if accelerator != "cpu" else torch.device("cpu")


def current_device_index() -> int:
    """Return the process-local accelerator index (zero after affinity masking)."""
    accelerator = require_accelerator()
    if accelerator == "cuda":
        return torch.cuda.current_device()
    return torch.xpu.current_device()


def empty_accelerator_cache() -> None:
    """Free cached memory on the active GPU backend, if one is visible."""
    accelerator = accelerator_name()
    if accelerator == "cuda":
        torch.cuda.empty_cache()
    elif accelerator == "xpu":
        torch.xpu.empty_cache()


def jax_platform() -> str:
    """Return the JAX platform corresponding to the selected Torch backend.

    Intel Extension for OpenXLA exposes Aurora GPUs to JAX as ``sycl``.
    """
    accelerator = require_accelerator()
    return "gpu" if accelerator == "cuda" else "sycl"


def get_jax_device(jax_module: Any, device_id: int | None = None) -> Any:
    """Select the JAX device paired with this process's visible accelerator."""
    platform = jax_platform()
    devices = jax_module.devices(platform)
    if not devices:
        raise RuntimeError(f"JAX reports no devices for platform {platform!r}")
    index = current_device_index() if device_id is None else device_id
    if index >= len(devices):
        raise RuntimeError(
            f"Requested JAX {platform} device {index}, but only {len(devices)} device(s) are visible. "
            "Set ZE_AFFINITY_MASK before launching an Aurora process."
        )
    return devices[index]


def lightning_trainer_kwargs() -> dict[str, Any]:
    """Return Lightning settings for one process-local accelerator.

    Lightning 2.5 does not yet ship an XPU accelerator.  The local shim uses
    upstream PyTorch XPU APIs and an explicit single-device strategy.
    """
    accelerator = require_accelerator()
    if accelerator == "xpu":
        from lightning.pytorch.strategies import SingleDeviceStrategy

        from proteinfoundation.utils.lightning_xpu import AuroraXPUAccelerator

        device = torch_device(require=True)
        return {
            "accelerator": AuroraXPUAccelerator(),
            "devices": 1,
            "strategy": SingleDeviceStrategy(device=device),
        }
    return {"accelerator": "gpu", "devices": 1}
