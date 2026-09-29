# Added for the Aurora XPU port; see aurora/README.md.
"""Small compute-node validation of the Proteina-Complexa Aurora port."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from proteinfoundation.utils.device_utils import lightning_trainer_kwargs, torch_device


def main() -> None:
    device = torch_device(require=True)
    assert device.type == "xpu", f"Aurora smoke expected XPU, got {device}"
    print(f"torch={torch.__version__}")
    print(f"device={device}")
    print(f"properties={torch.xpu.get_device_properties(device)}")

    lhs = torch.ones((256, 256), device=device)
    result = (lhs @ lhs).sum()
    torch.xpu.synchronize(device)
    assert result.item() == 16_777_216.0
    print(f"torch_xpu_matmul={result.item()}")

    # ALCF documents that PyG optional compiled extensions are CPU/CUDA-only.
    # The port's native-PyTorch replacement must therefore work on XPU.
    from proteinfoundation.nn.feature_factory.seq_cond_feats import scatter_mean

    reduced = scatter_mean(
        torch.tensor([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]], device=device),
        torch.tensor([0, 1, 0], device=device),
        dim_size=2,
    )
    torch.xpu.synchronize(device)
    assert torch.allclose(reduced.cpu(), torch.tensor([[5.0, 7.0], [5.0, 7.0]]))
    print("native_scatter_mean=passed")

    import jax
    import jax.numpy as jnp

    jax_devices = jax.devices("sycl")
    assert jax_devices, "Intel OpenXLA did not expose a SYCL JAX device"
    jax_result = jnp.ones((128, 128), dtype=jnp.float32) @ jnp.ones((128, 128), dtype=jnp.float32)
    assert float(jax_result.sum()) == 2_097_152.0
    print(f"jax_sycl_devices={jax_devices}")
    print(f"jax_sycl_matmul={float(jax_result.sum())}")

    from openmm import Platform

    platforms = [Platform.getPlatform(index).getName() for index in range(Platform.getNumPlatforms())]
    assert "OpenCL" in platforms, f"OpenMM OpenCL platform is unavailable: {platforms}"
    print(f"openmm_platforms={platforms}")

    import lightning as L

    class PredictModule(L.LightningModule):
        def predict_step(self, batch, batch_idx):
            return batch[0] * 2

    trainer = L.Trainer(
        **lightning_trainer_kwargs(),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    predictions = trainer.predict(PredictModule(), DataLoader(TensorDataset(torch.arange(4)), batch_size=2))
    assert torch.equal(torch.cat(predictions).cpu(), torch.tensor([0, 2, 4, 6]))
    print("lightning_xpu_predict=passed")
    print("Proteina-Complexa Aurora XPU smoke passed.")


if __name__ == "__main__":
    main()
