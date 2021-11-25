import torch.nn as nn
from .models import register


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


@register('Conv4')
class ConvNet4(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_dim = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x


@register('Conv4_21')
class CNNEncoder(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(CNNEncoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(x_dim, hid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hid_dim, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hid_dim, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hid_dim, affine=True),
            nn.LeakyReLU(0.2, True), )
        self.layer4 = nn.Sequential(
            nn.Conv2d(hid_dim, z_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(z_dim, affine=True),
            nn.LeakyReLU(0.2, True), )

        self.out_dim = z_dim * 21 * 21

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out
