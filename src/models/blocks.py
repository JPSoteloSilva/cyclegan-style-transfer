import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0
    ):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size, stride=stride, padding=padding
            ),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size, stride=stride, padding=padding
            ),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        normalize: bool = True,
        activation: str = "leaky_relu",
        negative_slope: float = 0.2,
        reflection_padding: bool = False,
        add_spectral_norm: bool = False,
        bias: bool = True,
    ):
        super(ConvBlock, self).__init__()
        layers = []
        if reflection_padding:
            layers.append(nn.ReflectionPad2d(in_channels))

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        if add_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        layers.append(conv)
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
        else:
            raise ValueError(f"Invalid activation function: {activation}")
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        super(UpsampleBlock, self).__init__()
        layers = [
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
