from typing import Literal, Optional

from pydantic import BaseModel, Field


class EarlyStoppingConfig(BaseModel):
    patience: int = Field(
        default=7,
        description="Number of epochs to wait before stopping if no improvement",
    )
    min_delta: float = Field(
        default=0.0,
        description="Minimum change in validation loss to qualify as an improvement",
    )


class TrainConfig(BaseModel):
    num_epochs: int = Field(default=20, description="Number of epochs to train")
    k: int = Field(
        default=1, description="Number of discriminator updates per generator update"
    )
    output_dir: str = Field(default="results/", description="Directory to save outputs")
    early_stopping: EarlyStoppingConfig = Field(
        default=EarlyStoppingConfig(),
        description="Early stopping configuration",
    )
    val_every: int = Field(default=2, description="Validate every n epochs")
    mifid_feature_size: Optional[int] = Field(
        default=None, description="Feature size for MiFID calculation"
    )
    loss_type: Literal["mse", "wgan"] = Field(
        default="mse", description="Type of GAN loss to use"
    )
    monitor_metric: Literal["loss", "mifid"] = Field(
        default="mifid", description="Metric to monitor for early stopping"
    )
    gp_weight: float = Field(
        default=10.0, description="Weight for gradient penalty in WGAN-GP"
    )


class DatasetConfig(BaseModel):
    data_dir: str = Field(
        default="data/", description="Directory containing the dataset"
    )
    image_size: int = Field(default=32, description="Size of input images")
    batch_size: int = Field(default=5, description="Batch size for training")
    pin_memory: bool = Field(default=False, description="Pin memory for data loading")
    dataset_num_workers: int = Field(
        default=0, description="Number of workers for data loading"
    )
    test_size: float = Field(
        default=0.2, description="Fraction of data used for testing"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    use_big_dataset: bool = Field(
        default=False, description="Flag to use larger dataset"
    )
    shuffle: bool = Field(default=True, description="Shuffle dataset during training")
    fraction_dataset: float = Field(
        default=1.0, description="Fraction of dataset to use"
    )


class GeneratorOptimizationConfig(BaseModel):
    lr: float = Field(default=0.0002, description="Learning rate")
    b1: float = Field(default=0.5, description="Beta1 parameter for Adam optimizer")
    b2: float = Field(default=0.999, description="Beta2 parameter for Adam optimizer")
    scheduler_type: Literal["step", "cosine", "plateau", "lambda"] = Field(
        default="step", description="Type of learning rate scheduler"
    )


class DiscriminatorOptimizationConfig(BaseModel):
    lr: float = Field(default=0.0001, description="Learning rate")
    b1: float = Field(default=0.5, description="Beta1 parameter for Adam optimizer")
    b2: float = Field(default=0.999, description="Beta2 parameter for Adam optimizer")
    scheduler_type: Literal["step", "cosine", "plateau", "lambda"] = Field(
        default="step", description="Type of learning rate scheduler"
    )


class OptimizationConfig(BaseModel):
    generator: GeneratorOptimizationConfig = Field(
        default=GeneratorOptimizationConfig(),
        description="Generator optimization configuration",
    )
    discriminator: DiscriminatorOptimizationConfig = Field(
        default=DiscriminatorOptimizationConfig(),
        description="Discriminator optimization configuration",
    )
    identity_loss_weight: float = Field(
        default=5.0, description="Weight for identity loss"
    )
    cycle_loss_weight: float = Field(
        default=10.0, description="Weight for cycle consistency loss"
    )
    gan_loss_weight: float = Field(default=1.0, description="Weight for GAN loss")


class ModelConfig(BaseModel):
    input_channels: int = Field(default=3, description="Number of input channels")
    channels: int = Field(default=60, description="Number of channels in the model")
    n_res_blocks: int = Field(default=9, description="Number of residual blocks")
    n_scale_blocks: int = Field(default=2, description="Number of scale blocks")
    discriminator_type: Literal["patchgan", "dcgan"] = Field(
        default="patchgan",
        description="Type of discriminator to use. 'patchgan' is the original CycleGAN architecture, 'dcgan' is better for WGAN loss",
    )
    image_size: int = Field(
        default=256, description="Size of input images (needed for DCGAN discriminator)"
    )


class Config(BaseModel):
    train: TrainConfig
    dataset: DatasetConfig
    optimization: OptimizationConfig
    model: ModelConfig
    device: str = Field(default="cuda", description="Device to run the model on")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from a YAML file."""
        import yaml

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
