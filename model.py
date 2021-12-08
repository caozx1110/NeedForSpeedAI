from torch import nn

class DriveNet(nn.Module):
    def __init__(self):
        super(DriveNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),  # 640 x 640
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(5, 5, 0),      # 128 x 128

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # 64 x 64
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # 32 x 32
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # 16 x 16

            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # 8 x 8

            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # 4 x 4
        )

        self.fc = nn.Sequential(  #fully connected layers
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 9),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


