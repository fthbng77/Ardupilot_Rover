import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SegNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SegNet, self).__init__()

        # Encoder layers
        self.encoder1 = nn.Sequential(
            DoubleConv(input_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        self.encoder2 = nn.Sequential(
            DoubleConv(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        self.encoder3 = nn.Sequential(
            DoubleConv(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        self.encoder4 = nn.Sequential(
            DoubleConv(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        self.encoder5 = nn.Sequential(
            DoubleConv(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )

        # Decoder layers
        self.decoder1 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            DoubleConv(512, 512)
        )
        self.decoder2 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            DoubleConv(512, 256)
        )
        self.decoder3 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            DoubleConv(256, 128)
        )
        self.decoder4 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            DoubleConv(128, 64)
        )
        self.decoder5 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            DoubleConv(64, num_classes)
        )

    def forward(self, x):
        x, idx1 = self.encoder1(x)
        x, idx2 = self.encoder2(x)
        x, idx3 = self.encoder3(x)
        x, idx4 = self.encoder4(x)
        x, idx5 = self.encoder5(x)

        x = self.decoder1[0](x, idx5)
        x = self.decoder1[1](x)
        x = self.decoder2[0](x, idx4)
        x = self.decoder2[1](x)
        x = self.decoder3[0](x, idx3)
        x = self.decoder3[1](x)
        x = self.decoder4[0](x, idx2)
        x = self.decoder4[1](x)
        x = self.decoder5[0](x, idx1)
        x = self.decoder5[1](x)


        return x

