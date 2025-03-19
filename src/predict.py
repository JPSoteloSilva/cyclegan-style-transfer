import os
from argparse import ArgumentParser

import torch

from src.config import Config
from src.models.generator import Generator
from src.utils import generate_images


def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/local.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    dataset_config = config.dataset
    model_config = config.model
    train_config = config.train

    # Initialize only the Generator BA
    G_BA = Generator(
        model_config.input_channels,
        model_config.channels,
        model_config.n_res_blocks,
        model_config.n_scale_blocks,
    )
    G_BA.to(config.device)

    # Load the model checkpoint
    checkpoint_path = args.model_path or os.path.join(
        train_config.output_dir, "best_model.pth"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location=config.device, weights_only=True
    )

    # Load only the Generator BA state dict
    G_BA.load_state_dict(checkpoint["G_BA_state_dict"])
    G_BA.eval()

    # Generate images
    generate_images(
        G_BA=G_BA,
        input_dir=dataset_config.data_dir + "/photo_jpg",
        output_dir=train_config.output_dir,
        batch_size=dataset_config.batch_size,
        device=config.device,
    )


if __name__ == "__main__":
    main()
