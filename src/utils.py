import torch
import torch.nn as nn

class ReLU(nn.Module):
    """The ResNet family uses ReLU as the activation function"""
    def __init__(self, x=None):
        super().__init__()
        # If called with ReLU(x) at initialization
        self.direct_x = x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0)

    def __call__(self, *args, **kwargs):
        # If called as direct function
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            return torch.clamp(args[0], min=0)
        return super().__call__(*args, **kwargs)


class LeakyReLU(nn.Module):
    """Also implemented the LeakyReLU from Scratch"""
    def __init__(self, negative_slope: float = 0.01, x=None):
        super().__init__()        
        if isinstance(negative_slope, torch.Tensor):
            self.direct_x = negative_slope
            self.negative_slope = 0.01
        else:
            self.direct_x = x
            self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0) + self.negative_slope * torch.clamp(x, max=0)

    def __call__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            return torch.clamp(args[0], min=0) + self.negative_slope * torch.clamp(args[0], max=0)
        return super().__call__(*args, **kwargs)