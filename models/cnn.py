import torch
from torch import nn

try:
    from basic import BasicModule
except ImportError:
    from .basic import BasicModule


class ShallowCNNModel(BasicModule):
    def __init__(self, num_classes=2):
        super(ShallowCNNModel, self).__init__()
        self.model_name = 'ShallowCNNModel'

        self.cov = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # (B, 16, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 16, 112, 112)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (B, 32, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 32, 56, 56)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 56, 56)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, 28, 28)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cov(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


class CNNModel(BasicModule):
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        self.model_name = 'CNNModel'

        # 特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 验证网络正确性
if __name__ == '__main__':
    classification = CNNModel()
    # 按照 batch_size=32，channel=3，size=224x224 输入
    inputs = torch.ones((32, 3, 224, 224))
    outputs = classification(inputs)
    print(outputs.shape)
