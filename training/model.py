import torch
from torch import nn


class MnistMLP(nn.Module):
    """
    MNIST multilayer perceptron:

        784 inputs -> 16 hidden neurons -> ReLU -> 10 logits
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(784, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # CrossEntropyLoss expects logits, so no softmax is needed.
        return x