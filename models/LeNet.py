import torch
import torch.nn as nn

from torchinfo import summary


class ModelConfig(object):
    """配置参数"""

    def __init__(self, notes=''):
        self.model_name = 'LeNet'
        self.save_path = f'./result/{self.model_name}_{notes}.ckpt'  # 模型训练结果
        self.log_path = './tf_log/' + self.model_name


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=1), nn.Sigmoid(),  # 31x28 ->29 26
            nn.AvgPool2d(kernel_size=2, stride=2),  # 29 26 14 13
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),  # 14 13->10 9
            nn.AvgPool2d(kernel_size=2, stride=2),  # 10 9 -> 5 4
            nn.Flatten(),
            nn.Linear(16 * 5 * 4, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.net(x)
        return x


if __name__ == '__main__':

    model = LeNet()
    print(model)
    input = torch.randn(8, 1, 31, 28)
    out = model(input)
    print(out.shape)

    batch_size = 256
    print(summary(model, input_size=(batch_size, 1, 31, 28), dtypes=[torch.float]))
