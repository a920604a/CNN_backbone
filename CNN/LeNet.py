import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self , num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16*6*6, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))  # torch.Size([16, 6, 32, 32])
        x = F.avg_pool2d(x, kernel_size=2, stride=2)  # torch.Size([16, 6, 16, 16])
        x = torch.sigmoid(self.conv2(x)) # torch.Size([16, 16, 12, 12])
        x = F.avg_pool2d(x, kernel_size=2, stride=2) # torch.Size([16, 16, 6, 6])
        x = torch.flatten(x, 1) # torch.Size([16, 576])
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
