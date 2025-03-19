import os
import random
from typing import Literal, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import (
    DiscriminatorOptimizationConfig,
    EarlyStoppingConfig,
    GeneratorOptimizationConfig,
    OptimizationConfig,
)
from src.dataset import DataLoader
from src.metrics import calculate_mifid
from src.models.discriminator import DCGANDiscriminator, PatchGANDiscriminator
from src.models.generator import Generator
from src.utils import add_scheduler, plot_losses_train, plot_mifid_scores, sample_images


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping if no improvement
            min_delta: Minimum change in validation loss to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Args:
            val_loss: Validation loss
        Returns:
            True if early stopping should be triggered, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop


class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image

        # Guardar la forma del batch de entrada para asegurar consistencia
        batch_size = image.size(0)
        returned_images = []

        for i in range(batch_size):
            img = image[i : i + 1]  # Obtener imagen individual del batch

            if len(self.images) < self.pool_size:
                self.images.append(img.clone())
                returned_images.append(img)
            else:
                if random.random() > 0.5:
                    # Usar imagen actual
                    returned_images.append(img)
                else:
                    # Usar imagen aleatoria del pool
                    idx = random.randint(0, len(self.images) - 1)
                    old_img = self.images[idx].clone()
                    self.images[idx] = img.clone()
                    returned_images.append(old_img)

        # Concatenar todas las imágenes en un solo tensor con el mismo batch_size que el original
        return torch.cat(returned_images, dim=0)


class Trainer:
    def __init__(
        self,
        G_AB: Generator,
        G_BA: Generator,
        D_A: Union[PatchGANDiscriminator, DCGANDiscriminator],
        D_B: Union[PatchGANDiscriminator, DCGANDiscriminator],
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimization_config: OptimizationConfig,
        early_stopping_config: EarlyStoppingConfig,
        device: str = "cuda",
        mifid_feature_size: int | None = None,
        num_epochs: int = 20,
        monitor_metric: Literal["loss", "mifid"] = "mifid",
        grad_clip_value: float = 1.0,
        loss_type: Literal["mse", "wgan"] = "mse",
        gp_weight: float = 10.0,
    ):
        """
        Args:
            G_AB: Generator A to B
            G_BA: Generator B to A
            D_A: Discriminator A
            D_B: Discriminator B
            train_loader: Training data loader
            val_loader: Validation data loader
            optimization_config: Optimization configuration
            early_stopping_config: Early stopping configuration
            device: Device to use for training
            mifid_feature_size: Feature size for MiFID. If None, MiFID will not be used.
            num_epochs: Number of epochs to train
            monitor_metric: Metric to monitor for early stopping and checkpointing
            grad_clip_value: Gradient clipping value
            loss_type: Type of GAN loss to use ('mse' or 'wgan')
            gp_weight: Gradient penalty weight
        """
        self.G_AB = G_AB
        self.G_BA = G_BA
        self.D_A = D_A
        self.D_B = D_B

        self.loss_type = loss_type
        if loss_type == "mse":
            self.criterion_GAN = nn.MSELoss()
        else:
            self.criterion_GAN = self.wasserstein_loss
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        self.num_epochs = num_epochs
        self.set_device(device)
        self.set_optimizers(
            generator_optimizer_config=optimization_config.generator,
            discriminator_optimizer_config=optimization_config.discriminator,
        )
        self.set_schedulers(
            g_scheduler_type=optimization_config.generator.scheduler_type,
            d_scheduler_type=optimization_config.discriminator.scheduler_type,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.identity_loss_weight = optimization_config.identity_loss_weight
        self.cycle_loss_weight = optimization_config.cycle_loss_weight
        self.gan_loss_weight = optimization_config.gan_loss_weight

        self.best_val_value = float("inf")

        self.early_stopping = EarlyStopping(
            patience=early_stopping_config.patience,
            min_delta=early_stopping_config.min_delta,
        )
        self.mifid_feature_size = mifid_feature_size
        self.monitor_metric = monitor_metric
        self.grad_clip_value = grad_clip_value
        self.gp_weight = gp_weight

        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

    def set_device(self, device: str = "cuda") -> None:
        """Set the device for the model and criterion."""
        self.G_AB.to(device)
        self.G_BA.to(device)
        self.D_A.to(device)
        self.D_B.to(device)
        if not self.loss_type == "wgan":
            self.criterion_GAN.to(device)
        self.criterion_cycle.to(device)
        self.criterion_identity.to(device)
        self.device = device

    def set_optimizers(
        self,
        generator_optimizer_config: GeneratorOptimizationConfig,
        discriminator_optimizer_config: DiscriminatorOptimizationConfig,
    ) -> None:
        """Set the optimizers for the model."""
        # Single optimizer for both generators
        self.optimizer_G = torch.optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=generator_optimizer_config.lr,
            betas=(generator_optimizer_config.b1, generator_optimizer_config.b2),
        )

        # Separate optimizers for discriminators (usually better to keep separate)
        self.optimizer_D_A = torch.optim.Adam(
            self.D_A.parameters(),
            lr=discriminator_optimizer_config.lr,
            betas=(
                discriminator_optimizer_config.b1,
                discriminator_optimizer_config.b2,
            ),
        )
        self.optimizer_D_B = torch.optim.Adam(
            self.D_B.parameters(),
            lr=discriminator_optimizer_config.lr,
            betas=(
                discriminator_optimizer_config.b1,
                discriminator_optimizer_config.b2,
            ),
        )

    def set_schedulers(
        self,
        g_scheduler_type: Literal["step", "cosine", "plateau", "lambda"] = "step",
        d_scheduler_type: Literal["step", "cosine", "plateau", "lambda"] = "step",
    ) -> None:
        """Set the schedulers for the model."""
        self.g_scheduler_type = g_scheduler_type
        self.d_scheduler_type = d_scheduler_type
        # Single scheduler for generator optimizer
        self.scheduler_G = add_scheduler(
            self.optimizer_G, g_scheduler_type, self.num_epochs
        )
        # Separate schedulers for discriminators
        self.scheduler_D_A = add_scheduler(
            self.optimizer_D_A, d_scheduler_type, self.num_epochs
        )
        self.scheduler_D_B = add_scheduler(
            self.optimizer_D_B, d_scheduler_type, self.num_epochs
        )

    def wasserstein_loss(self, y_pred, y_true):
        """
        Calculate Wasserstein loss.
        For real samples, y_true is 1, and we want to maximize the discriminator output
        For fake samples, y_true is -1, and we want to minimize the discriminator output
        """
        # Simply return the mean prediction multiplied by the expected label
        return -torch.mean(y_pred * y_true)

    def compute_generator_losses(self, real_A, real_B, fake_A, fake_B, valid):
        """Calculate all generator losses."""
        # Identity loss
        loss_id_A = self.criterion_identity(fake_B, real_A)
        loss_id_B = self.criterion_identity(fake_A, real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        if self.loss_type == "mse":
            loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
            loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)
        else:  # wgan
            # Get discriminator outputs
            fake_B_validity = self.D_B(fake_B)
            fake_A_validity = self.D_A(fake_A)

            # Handle PatchGAN outputs if needed
            if hasattr(fake_B_validity, "ndim") and fake_B_validity.ndim > 2:
                fake_B_validity = fake_B_validity.mean(dim=(2, 3))
            if hasattr(fake_A_validity, "ndim") and fake_A_validity.ndim > 2:
                fake_A_validity = fake_A_validity.mean(dim=(2, 3))

            # For WGAN we want to maximize discriminator output for generated samples
            loss_GAN_AB = -torch.mean(fake_B_validity)
            loss_GAN_BA = -torch.mean(fake_A_validity)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = self.G_BA(fake_B)
        recov_B = self.G_AB(fake_A)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total losses - separate for each generator
        loss_G_AB = (
            self.identity_loss_weight * loss_id_A
            + self.gan_loss_weight * loss_GAN_AB
            + self.cycle_loss_weight * loss_cycle_A
        )

        loss_G_BA = (
            self.identity_loss_weight * loss_id_B
            + self.gan_loss_weight * loss_GAN_BA
            + self.cycle_loss_weight * loss_cycle_B
        )

        return loss_G_AB, loss_G_BA, loss_identity, loss_GAN, loss_cycle

    def compute_discriminator_loss(self, D, real, fake, valid, fake_tensor):
        """
        Calculate discriminator loss for different architectures and loss types.

        Args:
            D: Discriminator model (PatchGAN or DCGAN)
            real: Real images
            fake: Fake images (detached from computation graph)
            valid: Target tensor for real images
            fake_tensor: Target tensor for fake images
        """
        # Obtain discriminator outputs
        real_output = D(real)
        fake_output = D(fake.detach())

        # Process based on loss type
        if self.loss_type == "mse":
            return self._compute_mse_discriminator_loss(
                real_output, fake_output, valid, fake_tensor
            )
        else:  # WGAN loss
            return self._compute_wgan_discriminator_loss(
                D, real, fake.detach(), real_output, fake_output
            )

    def _compute_mse_discriminator_loss(
        self, real_output, fake_output, valid, fake_tensor
    ):
        """Helper method to compute MSE-based discriminator loss."""
        # if len(real_output.shape) > 2:
        #     if valid.shape != real_output.shape:
        #         batch_size, _, h, w = real_output.size()
        #         # Reshape only if needed
        #         if len(valid.shape) == 2 or (
        #             valid.shape[2] == 1 and valid.shape[3] == 1
        #         ):
        #             valid = valid.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)
        #             fake_tensor = fake_tensor.view(batch_size, 1, 1, 1).expand(
        #                 batch_size, 1, h, w
        #             )

        # Calculate losses with potentially adjusted shapes
        loss_real = self.criterion_GAN(real_output, valid)
        loss_fake = self.criterion_GAN(fake_output, fake_tensor)

        # Return average loss
        return (loss_real + loss_fake) / 2

    def _compute_wgan_discriminator_loss(self, D, real, fake, real_output, fake_output):
        """Helper method to compute WGAN-based discriminator loss."""
        # If output has spatial dimensions, average to get scalar
        if len(real_output.shape) > 2:
            real_output = real_output.mean(dim=(2, 3))
            fake_output = fake_output.mean(dim=(2, 3))

        # Usar self.criterion_GAN (que es wasserstein_loss)
        real_labels = torch.ones_like(real_output).to(self.device)
        fake_labels = -torch.ones_like(fake_output).to(self.device)

        loss_real = self.criterion_GAN(real_output, real_labels)
        loss_fake = self.criterion_GAN(fake_output, fake_labels)

        # Gradient penalty (important for WGAN-GP)
        gradient_penalty = self.compute_gradient_penalty(D, real, fake)

        # Return combined loss
        return loss_real + loss_fake + self.gp_weight * gradient_penalty

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Compute gradient penalty for WGAN-GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(self.device)

        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        d_interpolates = D(interpolates)

        # Get gradients w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def validate(self) -> tuple[float, float | None, float | None, float]:
        """
        Perform validation and return the validation loss and MiFID score.

        Returns:
            tuple[float, float, float, float]: (validation_loss, mifid_A2B, mifid_B2A,
            monitor_value)
        """
        self.G_AB.eval()
        self.G_BA.eval()
        self.D_A.eval()
        self.D_B.eval()

        total_val_loss = 0
        all_real_A = []
        all_fake_B = []
        all_real_B = []
        all_fake_A = []

        for real_A, real_B in self.val_loader:
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            out_shape = [
                real_A.size(0),
                1,
                real_A.size(2) // self.D_A.scale_factor,
                real_A.size(3) // self.D_A.scale_factor,
            ]
            if self.loss_type == "mse":
                valid = torch.ones(out_shape).to(self.device)
                fake = torch.zeros(out_shape).to(self.device)
            else:  # wgan
                valid = torch.ones(out_shape).to(self.device)
                fake = -torch.ones(out_shape).to(self.device)

            # Generate fake samples
            with torch.no_grad():
                fake_B = self.G_AB(real_A)
                fake_A = self.G_BA(real_B)

                # Calculate generator losses
                loss_G_AB, loss_G_BA, _, _, _ = self.compute_generator_losses(
                    real_A, real_B, fake_A, fake_B, valid
                )

            # For WGAN, we need to allow gradients for gradient penalty computation
            if self.loss_type == "wgan":
                loss_D_A = self.compute_discriminator_loss(
                    self.D_A, real_A, fake_A, valid, fake
                )
                loss_D_B = self.compute_discriminator_loss(
                    self.D_B, real_B, fake_B, valid, fake
                )
            else:
                with torch.no_grad():
                    loss_D_A = self.compute_discriminator_loss(
                        self.D_A, real_A, fake_A, valid, fake
                    )
                    loss_D_B = self.compute_discriminator_loss(
                        self.D_B, real_B, fake_B, valid, fake
                    )

            # Move tensors to CPU before storing
            all_real_A.append(real_A.cpu())
            all_fake_B.append(fake_B.cpu())
            all_real_B.append(real_B.cpu())
            all_fake_A.append(fake_A.cpu())

            # Total validation loss combines generator and discriminator losses
            val_loss = (
                loss_G_AB
                + loss_G_BA  # Generator losses
                + loss_D_A
                + loss_D_B  # Discriminator losses
            ) / 4  # Average all losses

            total_val_loss += val_loss.item()

        # Concatenate on CPU to save memory
        real_A_all = torch.cat(all_real_A, dim=0).to(self.device)
        fake_B_all = torch.cat(all_fake_B, dim=0).to(self.device)
        real_B_all = torch.cat(all_real_B, dim=0).to(self.device)
        fake_A_all = torch.cat(all_fake_A, dim=0).to(self.device)

        # Calculate MiFID scores for both directions
        if self.mifid_feature_size is not None:
            mifid_A2B = calculate_mifid(
                real_A_all,
                fake_B_all,
                device=self.device,
                feature_size=self.mifid_feature_size,
            )
            mifid_B2A = calculate_mifid(
                real_B_all,
                fake_A_all,
                device=self.device,
                feature_size=self.mifid_feature_size,
            )
        else:
            mifid_A2B = None
            mifid_B2A = None

        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()

        # Add metric selection at the end
        monitor_value = (
            (mifid_A2B + mifid_B2A) / 2
            if self.monitor_metric == "mifid"
            and mifid_A2B is not None
            and mifid_B2A is not None
            else total_val_loss / len(self.val_loader)
        )
        return (
            total_val_loss / len(self.val_loader),
            mifid_A2B,
            mifid_B2A,
            monitor_value,
        )

    def save_checkpoint(
        self, val_value: float, epoch: int, output_dir: str = "", best: bool = True
    ) -> None:
        """Save model checkpoint if validation loss improves."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.best_val_value = val_value
        checkpoint = {
            "epoch": epoch,
            "G_AB_state_dict": self.G_AB.state_dict(),
            "G_BA_state_dict": self.G_BA.state_dict(),
            "D_A_state_dict": self.D_A.state_dict(),
            "D_B_state_dict": self.D_B.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_A_state_dict": self.optimizer_D_A.state_dict(),
            "optimizer_D_B_state_dict": self.optimizer_D_B.state_dict(),
            "val_value": val_value,
        }
        if best:
            torch.save(checkpoint, os.path.join(output_dir, "best_model.pth"))
        else:
            torch.save(checkpoint, os.path.join(output_dir, f"model_{epoch}.pth"))
        print(f"Saved checkpoint with validation value: {val_value:.4f}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint with error handling."""
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )

            # Validate checkpoint contents
            required_keys = [
                "G_AB_state_dict",
                "G_BA_state_dict",
                "D_A_state_dict",
                "D_B_state_dict",
                "optimizer_G_state_dict",
                "optimizer_D_A_state_dict",
                "optimizer_D_B_state_dict",
            ]

            if not all(key in checkpoint for key in required_keys):
                raise ValueError(f"Checkpoint missing required keys: {required_keys}")

            # Load states
            self.G_AB.load_state_dict(checkpoint["G_AB_state_dict"])
            self.G_BA.load_state_dict(checkpoint["G_BA_state_dict"])
            self.D_A.load_state_dict(checkpoint["D_A_state_dict"])
            self.D_B.load_state_dict(checkpoint["D_B_state_dict"])

            # Load optimizer states
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
            self.optimizer_D_A.load_state_dict(checkpoint["optimizer_D_A_state_dict"])
            self.optimizer_D_B.load_state_dict(checkpoint["optimizer_D_B_state_dict"])

        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {checkpoint_path}: {str(e)}"
            )

    def get_lr(self, optimizer):
        """Get current learning rate."""
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def train(self, k: int = 2, val_every: int = 2, output_dir: str = "") -> None:
        """
        Training loop with proper loss handling.
        For WGAN, k represents how many discriminator updates per generator update.
        For standard GAN, k=1 is typically used.
        """
        mifid_A2B_val_scores = []
        mifid_B2A_val_scores = []
        train_losses_G_AB = []
        train_losses_G_BA = []
        train_losses_D_A = []
        train_losses_D_B = []

        n_epochs_train = []
        n_epochs_val = []

        for epoch in tqdm(range(self.num_epochs)):
            self.G_AB.train()
            self.G_BA.train()
            self.D_A.train()
            self.D_B.train()

            # Track running losses for this epoch
            running_loss_D_A = 0.0
            running_loss_D_B = 0.0

            for i, (real_A, real_B) in enumerate(self.train_loader):
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)

                batch_size = real_A.size(0)
                out_shape = [
                    batch_size,
                    1,
                    real_A.size(2) // self.D_A.scale_factor,
                    real_A.size(3) // self.D_A.scale_factor,
                ]

                # Set up labels based on loss type
                if self.loss_type == "mse":
                    valid = torch.ones(out_shape).to(self.device)
                    fake = torch.zeros(out_shape).to(self.device)
                else:  # wgan
                    valid = torch.ones(out_shape).to(self.device)
                    fake = -torch.ones(out_shape).to(self.device)

                # ----------------
                # Train Discriminators
                # ----------------
                d_loss_A = 0
                d_loss_B = 0

                for _ in range(k):
                    self.optimizer_D_A.zero_grad()
                    self.optimizer_D_B.zero_grad()

                    # Generate fresh fake samples
                    with torch.no_grad():
                        fake_B = self.G_AB(real_A)
                        fake_A = self.G_BA(real_B)

                    # Usar el pool para entrenar discriminadores
                    fake_A_pool = self.fake_A_pool.query(fake_A.detach())
                    fake_B_pool = self.fake_B_pool.query(fake_B.detach())

                    # Train Discriminator A
                    loss_D_A = self.compute_discriminator_loss(
                        self.D_A, real_A, fake_A_pool, valid, fake
                    )
                    loss_D_A.backward()
                    d_loss_A += loss_D_A.item()

                    # Train Discriminator B
                    loss_D_B = self.compute_discriminator_loss(
                        self.D_B, real_B, fake_B_pool, valid, fake
                    )
                    loss_D_B.backward()
                    d_loss_B += loss_D_B.item()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.D_A.parameters(), self.grad_clip_value
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.D_B.parameters(), self.grad_clip_value
                    )

                    self.optimizer_D_A.step()
                    self.optimizer_D_B.step()

                    if self.loss_type == "wgan":
                        for d in [self.D_A, self.D_B]:
                            for p in d.parameters():
                                p.data.clamp_(-0.01, 0.01)

                # Average discriminator losses over k steps
                d_loss_A /= k
                d_loss_B /= k
                running_loss_D_A += d_loss_A
                running_loss_D_B += d_loss_B

                # ----------------
                # Train Generators
                # ----------------
                self.optimizer_G.zero_grad()

                # Generate fake samples
                fake_B = self.G_AB(real_A)
                fake_A = self.G_BA(real_B)

                # Compute generator losses
                loss_G_AB, loss_G_BA, loss_identity, loss_GAN, loss_cycle = (
                    self.compute_generator_losses(real_A, real_B, fake_A, fake_B, valid)
                )

                # Combine generator losses and do a single backward pass
                total_G_loss = loss_G_AB + loss_G_BA
                total_G_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.G_AB.parameters(), self.grad_clip_value
                )
                torch.nn.utils.clip_grad_norm_(
                    self.G_BA.parameters(), self.grad_clip_value
                )

                self.optimizer_G.step()

                # Clear memory after each batch
                torch.cuda.empty_cache()

            # Update schedulers
            if self.g_scheduler_type == "plateau":
                self.scheduler_G.step(total_G_loss)
            else:
                self.scheduler_G.step()  # type: ignore

            if self.d_scheduler_type == "plateau":
                self.scheduler_D_A.step(d_loss_A)  # type: ignore
                self.scheduler_D_B.step(d_loss_B)  # type: ignore
            else:
                self.scheduler_D_A.step()  # type: ignore
                self.scheduler_D_B.step()  # type: ignore

            # Log losses
            train_losses_G_AB.append(loss_G_AB.item())
            train_losses_G_BA.append(loss_G_BA.item())
            train_losses_D_A.append(running_loss_D_A / len(self.train_loader))
            train_losses_D_B.append(running_loss_D_B / len(self.train_loader))
            n_epochs_train.append(epoch + 1)

            if (epoch + 1) % val_every == 0:
                val_loss, mifid_A2B_val, mifid_B2A_val, monitor_value = self.validate()
                mifid_A2B_val_scores.append(mifid_A2B_val)
                mifid_B2A_val_scores.append(mifid_B2A_val)
                n_epochs_val.append(epoch + 1)
                if monitor_value < self.best_val_value:
                    self.save_checkpoint(monitor_value, epoch, output_dir)
                else:
                    self.save_checkpoint(
                        self.best_val_value, epoch, output_dir, best=False
                    )

                if self.early_stopping(monitor_value):
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
                loss_D = (loss_D_A + loss_D_B) / 2  # type: ignore
                print(f"[Epoch {epoch+1}/{self.num_epochs}]")
                print(
                    f"[G_AB loss: {loss_G_AB:.4f} | G_BA loss: {loss_G_BA:.4f} | identity: {loss_identity:.4f} GAN: {loss_GAN:.4f} cycle: {loss_cycle:.4f}]"
                )
                print(
                    f"[D loss: {loss_D.item():.4f} | D_A: {loss_D_A.item():.4f} D_B: {loss_D_B.item():.4f}]"
                )
                print(f"[Validation loss: {val_loss:.4f}]")
                if self.mifid_feature_size is not None:
                    print(f"[MiFID score: {mifid_A2B_val:.4f} | {mifid_B2A_val:.4f}]")
                sample_images(
                    real_A,
                    real_B,
                    self.G_AB,
                    self.G_BA,
                    figside=2.5,
                    output_dir=output_dir,
                    epoch=epoch,
                    device=self.device,
                )

                # Log learning rates
                print(
                    f"Learning rates - G_AB: {self.get_lr(self.optimizer_G):.6f}, "
                    f"G_BA: {self.get_lr(self.optimizer_G):.6f}, "
                    f"D_A: {self.get_lr(self.optimizer_D_A):.6f}, "
                    f"D_B: {self.get_lr(self.optimizer_D_B):.6f}"
                )
        if self.mifid_feature_size is not None:
            plot_mifid_scores(
                mifid_A2B_val_scores,
                mifid_B2A_val_scores,
                n_epochs_val,
                output_dir,
            )
        plot_losses_train(
            train_losses_G_AB,
            train_losses_G_BA,
            train_losses_D_A,
            train_losses_D_B,
            output_dir,
        )
