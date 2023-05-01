import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, outputs):
        super(CNN, self).__init__()

        self.selectTrackFeatures = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.Softplus(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.Softplus(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.Softplus()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(9216, 16),
            nn.Linear(16, outputs)
        )

    def forward(self, state):
        x = self.selectTrackFeatures(state)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
