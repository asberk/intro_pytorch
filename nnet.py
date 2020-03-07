import torch
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, inputs):
        out = torch.relu(self.fc1(inputs))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = self.fc4(out)
        return out
