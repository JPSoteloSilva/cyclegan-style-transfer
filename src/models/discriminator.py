import torch.nn as nn

from src.models.blocks import ConvBlock


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator implementation.
    This is the original discriminator architecture used in CycleGAN.
    It classifies whether image patches are real or fake, rather than the entire image.
    """

    def __init__(self, in_channels: int):
        super(PatchGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            ConvBlock(in_channels, 64, 4, 2, 1),
            ConvBlock(64, 128, 4, 2, 1),
            ConvBlock(128, 256, 4, 2, 1),
            ConvBlock(256, 512, 4, 2, 1),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
        )

        self.scale_factor = 16

    def forward(self, x):
        return self.model(x)


class DCGANDiscriminator(nn.Module):
    """
    DCGAN-style discriminator.
    This architecture is better suited for WGAN loss as it outputs a single value
    for the entire image, rather than a patch-based output.
    """

    def __init__(self, in_channels: int, img_size: int = 256):
        super(DCGANDiscriminator, self).__init__()

        # Calculate final output size based on input size
        self.feature_size = img_size // 16  # After 4 downsamplings with stride 2

        self.model = nn.Sequential(
            # First layer: No normalization as per WGAN paper recommendation
            ConvBlock(
                in_channels,
                64,
                4,
                2,
                1,
                normalize=False,
                activation="leaky_relu",
                negative_slope=0.2,
                add_spectral_norm=False,
                bias=False,
            ),
            ConvBlock(
                64,
                128,
                4,
                2,
                1,
                normalize=True,
                activation="leaky_relu",
                negative_slope=0.2,
                add_spectral_norm=False,
                bias=False,
            ),
            ConvBlock(
                128,
                256,
                4,
                2,
                1,
                normalize=True,
                activation="leaky_relu",
                negative_slope=0.2,
                add_spectral_norm=False,
                bias=False,
            ),
            ConvBlock(
                256,
                512,
                4,
                2,
                1,
                normalize=True,
                activation="leaky_relu",
                negative_slope=0.2,
                add_spectral_norm=False,
                bias=False,
            ),
        )

        # Final layer to scalar output
        self.final_layer = nn.Conv2d(512, 1, self.feature_size, bias=False)

        self.scale_factor = img_size

    def forward(self, x):
        features = self.model(x)
        output = self.final_layer(features)
        return output.view(x.size(0), -1)  # Flatten to [batch_size, 1]


# Define a factory function to create the appropriate discriminator
def Discriminator(in_channels: int, disc_type: str = "patchgan", img_size: int = 256):
    """
    Factory function to create a discriminator of the specified type.

    Args:
        in_channels: Number of input channels
        disc_type: Type of discriminator to use ('patchgan' or 'dcgan')
        img_size: Size of input images (for DCGAN)

    Returns:
        A discriminator model
    """
    if disc_type.lower() == "patchgan":
        return PatchGANDiscriminator(in_channels)
    elif disc_type.lower() == "dcgan":
        return DCGANDiscriminator(in_channels, img_size)
    else:
        raise ValueError(f"Unknown discriminator type: {disc_type}")
