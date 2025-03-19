import os
import shutil
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    ReduceLROnPlateau,
    StepLR,
)
from torchvision.utils import make_grid
from tqdm import tqdm

from src.models import Generator


def add_scheduler(
    optimizer: Optimizer,
    scheduler_type: Literal["step", "cosine", "plateau", "lambda"],
    num_epochs: int = 20,
) -> Union[StepLR, CosineAnnealingLR, ReduceLROnPlateau, LambdaLR]:
    if scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=int(num_epochs / 10), gamma=0.1)
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(num_epochs / 10)
        )
    elif scheduler_type == "lambda":
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1 - max(0, epoch - 10) / (num_epochs - 10),
        )
    return scheduler


def sample_images(
    real_A: Tensor,
    real_B: Tensor,
    G_AB: Generator,
    G_BA: Generator,
    figside: float = 2.5,
    output_dir: str = "",
    epoch: int = 0,
    device: str = "cuda",
) -> None:
    output_dir = output_dir + "/sample_images/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G_AB.to(device)
    G_BA.to(device)
    G_AB.eval()
    G_BA.eval()

    real_A = real_A.to(device)
    fake_B = G_AB(real_A).detach()
    real_B = real_B.to(device)
    fake_A = G_BA(real_B).detach()

    nrows = real_A.size(0)
    real_A = make_grid(real_A, nrow=nrows, normalize=True)
    fake_B = make_grid(fake_B, nrow=nrows, normalize=True)
    real_B = make_grid(real_B, nrow=nrows, normalize=True)
    fake_A = make_grid(fake_A, nrow=nrows, normalize=True)

    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1).cpu().permute(1, 2, 0)

    plt.figure(figsize=(figside * nrows, figside * 4))
    plt.imshow(image_grid)
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, f"{epoch}.png"))
    plt.show()
    plt.close()


def generate_images(
    G_BA: Generator,
    input_dir: str,
    output_dir: str,
    batch_size: int = 4,
    transforms: Optional[T.Compose] = None,
    device: str = "cuda",
) -> None:
    """Generate and save images using a trained generator.

    Args:
        G_BA: Generator model to use for image generation
        input_dir: Directory containing input images
        output_dir: Directory to save generated images
        device: Device to run the model on
        batch_size: Batch size for processing
        transforms: Optional transforms to apply to input images
    """
    images_dir = output_dir + "/images/"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    if transforms is None:
        transforms = T.Compose(
            [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    to_image = T.ToPILImage()

    files = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    G_BA.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(files), batch_size), desc="Generating images"):
            imgs = []
            for j in range(i, min(len(files), i + batch_size)):
                img = Image.open(files[j])
                img = transforms(img)
                imgs.append(img)
            imgs = torch.stack(imgs, 0).to(device)

            # Generate images
            fake_imgs = G_BA(imgs).detach().cpu()

            # Save generated images
            for j in range(fake_imgs.size(0)):
                img = fake_imgs[j].squeeze().permute(1, 2, 0)
                img_arr = img.numpy()
                img_arr = (
                    (img_arr - np.min(img_arr))
                    * 255
                    / (np.max(img_arr) - np.min(img_arr))
                )
                img_arr = img_arr.astype(np.uint8)

                img = to_image(img_arr)
                _, name = os.path.split(files[i + j])
                img.save(os.path.join(images_dir, name))

    # Create a zip file of the generated images at the same level as images directory
    shutil.make_archive(os.path.join(output_dir, "images"), "zip", images_dir)


def plot_mifid_scores(
    mifid_A2B_val: list[float],
    mifid_B2A_val: list[float],
    n_epochs_val: list[int],
    output_dir: str,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(n_epochs_val, mifid_A2B_val, label="MiFID A2B Val")
    plt.plot(n_epochs_val, mifid_B2A_val, label="MiFID B2A Val")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "mifid_scores.png"))
    plt.show()


def plot_losses_train(
    loss_G_AB: list[float],
    loss_G_BA: list[float],
    loss_D_A: list[float],
    loss_D_B: list[float],
    output_dir: str,
) -> None:
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.plot(loss_G_AB, label="Loss G AB")
    plt.plot(loss_G_BA, label="Loss G BA")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(loss_D_A, label="Loss D A")
    plt.plot(loss_D_B, label="Loss D B")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "losses_G_D_train.png"))
    plt.show()
