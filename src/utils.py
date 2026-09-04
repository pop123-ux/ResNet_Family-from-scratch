import torch
import torch.nn as nn

class ReLU(nn.Module):
    """The ResNet family uses ReLU (Rectified Linear Unit) as the activation function"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0) # Fixes the minimum value to 0

class LeakyReLU(nn.Module):
    """Also implemented the LeakyReLU from Scratch"""
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0) + self.negative_slope * torch.clamp(x, max=0)