import torch
from torch import nn

class MnistClassifier(nn.Module):
    """This class defines a basic convolutional neural nework for training the MNIST task"""

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=3),
            nn.LeakyReLU(),
        )
        self.flatten = nn.Flatten(start_dim=-3, end_dim=-1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=20736, out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-3) / 256
        x = x.to(torch.float32)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x
