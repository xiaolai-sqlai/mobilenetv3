import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class hSigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3) / 6
        return out


class hSwish(nn.Module):
    def forward(self, x):
        out = x * (F.relu6(x + 3) / 6)
        return out


class SEblock(nn.Module):

    def __init__(self, in_size, reduction=4):
        super(SEblock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),  
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hSigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):

    def __init__(self, kernel_size, in_channels, out_channels, stride, expand_channels, semodule, nonlinear):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=expand_channels, kernel_size=1, stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(expand_channels)
        self.nl1 = nonlinear
        self.convDW = nn.Conv2d(in_channels=expand_channels, out_channels=expand_channels, kernel_size=kernel_size,padding=kernel_size//2,stride=stride, groups=expand_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_channels)
        self.nl2 = nonlinear
        self.conv3 = nn.Conv2d(in_channels=expand_channels, out_channels=out_channels, kernel_size=1, stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.nl3 = nonlinear
        self.shortcut = nn.Sequential()
        
        if (stride ==1 and in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.nl1(self.bn1(self.conv1(x)))
        out = self.nl2(self.bn2(self.convDW(out)))
        if self.se is not None:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.stride == 1:
            out = self.nl3(out + self.shortcut(x))
        else:
            out = self.nl3(out)
        return out


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV3Large, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.nl1 = hSwish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 1, 16, None, nn.ReLU6(inplace=True)),
            Block(3, 16, 24, 2, 64, None, nn.ReLU6(inplace=True)),
            Block(3, 24, 24, 1, 72, None, nn.ReLU6(inplace=True)),
            Block(5, 24, 40, 2, 72, SEblock(in_size=72), nn.ReLU6(inplace=True)),
            Block(5, 40, 40, 1, 120, SEblock(in_size=120), nn.ReLU6(inplace=True)),
            Block(5, 40, 40, 1, 120, SEblock(in_size=120), nn.ReLU6(inplace=True)),
            Block(3, 40, 80, 2, 240, None, hSwish()),
            Block(3, 80, 80, 1, 200, None, hSwish()),
            Block(3, 80, 80, 1, 184, None, hSwish()),
            Block(3, 80, 80, 1, 184, None, hSwish()),
            Block(3, 80, 112, 1, 480, SEblock(in_size=480), hSwish()),
            Block(3, 112, 112, 1, 672, SEblock(in_size=672), hSwish()),
            Block(5, 112, 160, 2, 672, SEblock(in_size=672), hSwish()),
            Block(5, 160, 160, 1, 960, SEblock(in_size=960), hSwish()),
            Block(5, 160, 160, 1, 960, SEblock(in_size=960), hSwish())
        )
        self.conv2 = nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.nl2 = hSwish()
        self.pool = nn.AdaptiveAvgPool2d(1)  
        self.conv3 = nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, bias=False)
        self.nl3 = hSwish()
        self.conv4 = nn.Conv2d(in_channels=1280, out_channels=num_classes, kernel_size=1, bias=False)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.nl1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.nl2(self.bn2(self.conv2(out)))
        out = self.nl3(self.conv3(self.pool(out)))
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        return out
