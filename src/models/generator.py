import torch.nn as nn

from src.models.blocks import ConvBlock, ResidualBlock, UpsampleBlock


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int = 64,
        num_residual_blocks: int = 9,
        num_scale_blocks: int = 2,
    ):
        super(Generator, self).__init__()

        self.conv = ConvBlock(
            in_channels,
            channels,
            kernel_size=2 * in_channels + 1,
            reflection_padding=True,
            activation="relu",
        )

        down = []
        for _ in range(num_scale_blocks):
            out_channels = channels * 2
            down += [
                ConvBlock(
                    channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    activation="relu",
                )
            ]
            channels = out_channels
        self.down = nn.Sequential(*down)

        trans = []
        for _ in range(num_residual_blocks):
            trans += [ResidualBlock(channels, padding=0)]
        self.trans = nn.Sequential(*trans)

        up = []
        for _ in range(num_scale_blocks):
            out_channels = channels // 2
            up += [
                UpsampleBlock(
                    channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            ]
            channels = out_channels
        self.up = nn.Sequential(*up)

        self.out = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(channels, in_channels, 2 * in_channels + 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        x = self.trans(x)
        x = self.up(x)
        x = self.out(x)
        return x
