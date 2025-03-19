from typing import Union

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance


def calculate_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: Union[str, torch.device] = "cuda",
) -> float:
    """
    Calculate the Fréchet Inception Distance (FID) between real and generated images.

    Args:
        real_images: Tensor of real images, shape (N, C, H, W), normalized to [-1, 1]
            or [0, 1]
        generated_images: Tensor of generated images, shape (N, C, H, W), normalized to
            [-1, 1] or [0, 1]
        device: Device to run the calculation on

    Returns:
        float: The calculated FID score (lower is better)
    """
    # Initialize FID metric
    fid = FrechetInceptionDistance(normalize=True).to(device)

    # Make sure images are in the right format (float32) and on the correct device
    real_images = real_images.to(device, dtype=torch.float32)
    generated_images = generated_images.to(device, dtype=torch.float32)

    # Update FID with real and generated images
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)

    # Calculate FID
    fid_score = fid.compute()

    return float(fid_score.cpu())


def calculate_mifid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: Union[str, torch.device] = "cuda",
    feature_size: int = 768,
) -> float:
    """
    Calculate the Memorization-Informed Frechet Inception Distance (MiFID) between real and generated images.
    MiFID penalizes memorization of the training set by the generator.

    Args:
        real_images: Tensor of real images, shape (N, C, H, W), normalized to [-1, 1]
            or [0, 1]
        generated_images: Tensor of generated images, shape (N, C, H, W), normalized to
            [-1, 1] or [0, 1]
        device: Device to run the calculation on
        feature_size: Size of the feature layer from InceptionV3 (64, 192, 768, or 2048)

    Returns:
        float: The calculated MiFID score (lower is better)
    """
    # Initialize MiFID metric
    mifid = MemorizationInformedFrechetInceptionDistance(
        feature=feature_size,
        normalize=True,
        reset_real_features=True,
    ).to(device)

    # Make sure images are in the right format (float32) and on the correct device
    real_images = real_images.to(device, dtype=torch.float32)
    generated_images = generated_images.to(device, dtype=torch.float32)

    # Update MiFID with real and generated images
    mifid.update(real_images, real=True)
    mifid.update(generated_images, real=False)

    # Calculate MiFID
    mifid_score = mifid.compute()

    return float(mifid_score.cpu())
