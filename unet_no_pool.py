"""
U-Net class without max pooling layers after the double convolution blocks.
The blocks containing the double convolutions were changed to add a stride=2 and a padding to the second convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetNoPool(nn.Module):
    def __init__(self):
        super().__init__()

        # Contraction path : Encoder
        self.c1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=(3, 3), padding="same"
        )  # 128x128x16
        self.p1 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(2, 2),
        )  # 64x64x16
        self.c2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same"
        )  # 64x64x32
        self.p2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(2, 2),
        )  # 32x32x32
        self.c3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"
        )  # 32x32x64
        self.p3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(2, 2),
        )  # 16x16x64
        self.c4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"
        )  # 16x16x128
        self.p4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(2, 2),
        )  # 8x8x128
        self.c5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same"
        )  # 8x8x256

        # Expansion path : Decoder
        # upsampling
        self.u6 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(2, 2),
            stride=(2, 2),
        )  # 16x16x128 -> skip connection u6+c4 : 16x16x256
        self.c6 = doubleConv2d(
            in_channels=256, out_channels=128, kernel_size=(3, 3), padding="same"
        )  # 16x16x128
        self.u7 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(2, 2),
            stride=(2, 2),
        )  # 32x32x64 -> skip connection u7+c3 : 32x32x128
        self.c7 = doubleConv2d(
            in_channels=128, out_channels=64, kernel_size=(3, 3), padding="same"
        )  # 32x32x64
        self.u8 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(2, 2),
            stride=(2, 2),
        )  # 64x64x32 -> skip connection u8+c2 : 64x64x64
        self.c8 = doubleConv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), padding="same"
        )  # 64x64x32
        self.u9 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=(2, 2),
            stride=(2, 2),
        )  # 128x128x16 -> skip connection u9+c1 : 128x128x32
        self.c9 = doubleConv2d(
            in_channels=32, out_channels=16, kernel_size=(3, 3), padding="same"
        )  # 128x128x16
        self.c10 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=(1, 1)
        )  # 128x128x1

    def forward(self, x):

        # Encoder
        c1 = F.relu(self.c1(x))
        p1 = F.relu(self.p1(c1))
        c2 = F.relu(self.c2(p1))
        p2 = F.relu(self.p2(c2))
        c3 = F.relu(self.c3(p2))
        p3 = F.relu(self.p3(c3))
        c4 = F.relu(self.c4(p3))
        p4 = F.relu(self.p4(c4))
        c5 = F.relu(self.c5(p4))

        # Decoder
        u6 = self.u6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        u7 = self.u7(self.c6(u6))
        u7 = torch.cat([u7, c3], dim=1)
        u8 = self.u8(self.c7(u7))
        u8 = torch.cat([u8, c2], dim=1)
        u9 = self.u9(self.c8(u8))
        u9 = torch.cat([u9, c1], dim=1)
        c10 = self.c10(self.c9(u9))
        ret = torch.sigmoid(c10)

        return ret


class doubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))))
