"""
@function: 分类网络结构
"""
from torch import nn

class DriveNet(nn.Module):
    def __init__(self):
        """
        This net is used for classification, the input image size is 480 * 640.
        """
        super(DriveNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5)),
            # shape: 96 * 128
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # shape: 48 * 64
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # shape: 24 * 32
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # shape: 12 * 16
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # shape: 6 * 8
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # shape: 3 * 4
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 3 * 4, 6),
            # nn.ReLU(True),
            # nn.Linear(512, 5),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(-1, 32 * 3 * 4)
        out = self.fc(out)
        return out
