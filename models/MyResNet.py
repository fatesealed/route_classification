import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class Residual(nn.Module):  # @save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class MyResNet(nn.Module):
    def __init__(self, classes=10):
        super(MyResNet, self).__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=3, dilation=2),
                                nn.BatchNorm2d(32), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(*resnet_block(32, 32, 2, first_block=True))
        self.b3 = nn.Sequential(*resnet_block(32, 64, 2))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.avg(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = MyResNet()
    print(model)
    input = torch.randn(8, 1, 31, 28)
    out = model(input)
    print(out.shape)

    batch_size = 256
    print(summary(model, input_size=(batch_size, 1, 31, 28), dtypes=[torch.float]))
