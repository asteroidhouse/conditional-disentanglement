"""Simple models.
"""
import pdb
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x.reshape(x.shape[0], -1)  # Flatten x
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class SimpleConvNet(nn.Module):
  def __init__(self, output_dim=10, batch_norm=False):
    super(SimpleConvNet, self).__init__()

    self.batch_norm = batch_norm

    if batch_norm:
      self.layer1 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=5, padding=2),
        # nn.Conv2d(1, 16, kernel_size=5, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2))
      self.layer2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=5, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2))
    else:
      self.layer1 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=5, padding=2),
        # nn.Conv2d(1, 16, kernel_size=5, padding=2),
        # nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2))
      self.layer2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=5, padding=2),
        # nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2))

    self.fc = nn.Linear(2*2048, output_dim)

  def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
