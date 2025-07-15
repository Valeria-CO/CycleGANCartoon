import torch
import torch.nn as nn


class GenCartoon(nn.Module):
    def __init__(self):
        super().__init__()

        self.res_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(256)
        )

        self.input = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3,64, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.encod = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decod = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.input(x)
        x = self.encod(x)
        for _ in range(9):
            x = x + self.res_block(x)
        x = self.decod(x)
        x = self.output(x)
        return x


class DisCartoon(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),                     # (B, 64, H/2, W/2)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),                     # (B, 128, H/4, W/4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),                      # (B, 256, H/8, W/8)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),                      # (B, 512, H/16, W/16)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # (B, 1, H/16 - 1, W/16 - 1)
        )

    def forward(self, x):
        return self.model(x)


# model = GenCartoon()
# dummy = torch.randn(1, 3, 128, 128)  # подаём случайное изображение
# out = model(dummy)
# print(out.shape)  # должно быть torch.Size([1, 3, 128, 128])
#
#
# D = DisCartoon()
# dummy = torch.randn(1, 3, 128, 128)
# out = D(dummy)
# print(out.shape)
