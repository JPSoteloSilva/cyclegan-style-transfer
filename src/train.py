from argparse import ArgumentParser

from src.config import Config
from src.dataset import create_dataloaders
from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.trainer import Trainer


def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/local.yaml")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    train_config = config.train
    dataset_config = config.dataset
    optimization_config = config.optimization
    model_config = config.model

    # Log discriminator type for clarity
    disc_type = model_config.discriminator_type
    print(f"Using {disc_type} discriminator architecture")
    print(f"Using {train_config.loss_type} loss type")

    # Initialize models
    G_AB = Generator(
        model_config.input_channels,
        model_config.channels,
        model_config.n_res_blocks,
        model_config.n_scale_blocks,
    )
    G_BA = Generator(
        model_config.input_channels,
        model_config.channels,
        model_config.n_res_blocks,
        model_config.n_scale_blocks,
    )

    # Create discriminators with the specified type from config
    D_A = Discriminator(
        in_channels=model_config.input_channels,
        disc_type=model_config.discriminator_type,
        img_size=model_config.image_size,
    )
    D_B = Discriminator(
        in_channels=model_config.input_channels,
        disc_type=model_config.discriminator_type,
        img_size=model_config.image_size,
    )

    train_loader, val_loader = create_dataloaders(
        data_dir=dataset_config.data_dir,
        batch_size=dataset_config.batch_size,
        test_size=dataset_config.test_size,
        num_workers=dataset_config.dataset_num_workers,
        pin_memory=dataset_config.pin_memory,
        seed=dataset_config.seed,
        use_big_dataset=dataset_config.use_big_dataset,
        shuffle=dataset_config.shuffle,
    )

    # Initialize trainer
    trainer = Trainer(
        G_AB=G_AB,
        G_BA=G_BA,
        D_A=D_A,
        D_B=D_B,
        train_loader=train_loader,
        val_loader=val_loader,
        optimization_config=optimization_config,
        early_stopping_config=train_config.early_stopping,
        device=config.device,
        mifid_feature_size=train_config.mifid_feature_size,
        monitor_metric=train_config.monitor_metric,
        num_epochs=train_config.num_epochs,
        loss_type=train_config.loss_type,
        gp_weight=train_config.gp_weight,
    )

    # Start training
    trainer.train(
        k=train_config.k,
        val_every=train_config.val_every,
        output_dir=train_config.output_dir,
    )


if __name__ == "__main__":
    main()
