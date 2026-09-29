# Added for the Aurora XPU port; see aurora/README.md.
"""Minimal Lightning accelerator for Intel GPUs.

This is intentionally a small compatibility shim for Lightning 2.5.  Aurora's
``frameworks`` module already supplies upstream PyTorch XPU support, so no
CUDA compatibility layer or deprecated IPEX import is needed.
"""

from __future__ import annotations

from typing import Any

import torch
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class AuroraXPUAccelerator(Accelerator):
    """Lightning single-device accelerator backed by ``torch.xpu``."""

    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise MisconfigurationException(f"Device should be XPU, got {device} instead.")
        torch.xpu.set_device(device)

    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        return torch.xpu.memory_stats(device)

    def teardown(self) -> None:
        if torch.xpu.is_available():
            torch.xpu.empty_cache()

    @staticmethod
    def parse_devices(devices: int | str | list[int]) -> list[int]:
        if isinstance(devices, int):
            if devices < 1:
                raise MisconfigurationException("XPU devices must be a positive integer.")
            return list(range(devices))
        if isinstance(devices, str):
            if devices == "auto":
                return list(range(AuroraXPUAccelerator.auto_device_count()))
            try:
                return [int(device.strip()) for device in devices.split(",")]
            except ValueError as exc:
                raise MisconfigurationException(f"Invalid XPU device specification: {devices!r}") from exc
        if not devices:
            raise MisconfigurationException("At least one XPU device must be selected.")
        return devices

    @staticmethod
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        return [torch.device("xpu", device) for device in devices]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        return bool(hasattr(torch, "xpu") and torch.xpu.is_available())

    @staticmethod
    def name() -> str:
        return "xpu"

    @classmethod
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(cls.name(), cls, description=cls.__name__)
