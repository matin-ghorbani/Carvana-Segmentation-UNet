import torch
from torch import nn
import torchvision.transforms.functional as tf


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # It is not in original paper
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # It is not in original paper
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=1,  # In the original paper they use 2. But we use 1 for binary image segmentation
            features=[64, 128, 256, 512]
    ):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(
                in_channels=feature * 2, out_channels=feature, kernel_size=2, stride=2
            ))
            self.ups.append(DoubleConv(
                feature * 2, feature
            ))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse the skip_connections list

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                # In original paper they do croping
                # (N, num_channels, H, W)
                x = tf.resize(x, size=skip_connection.shape[2:])

            skip_concat = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](skip_concat)

        return self.final_conv(x)


def main() -> None:
    x = torch.randn((3, 1, 161, 161))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)

    print(f'{x.shape = }')
    print(f'{preds.shape = }')

    assert preds.shape == x.shape


if __name__ == "__main__":
    main()
