import time
import os
import torch
import torch.nn as nn
from src.utils import ReLU, LeakyReLU

EPOCHS: int = 30

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, device=None):
        super().__init__()
        # Trainable params
        self.weight = nn.Parameter(torch.ones(num_features)) # The gamma
        self.bias = nn.Parameter(torch.zeros(num_features)) # The beta
        
        self.eps = eps
        self.momentum = momentum
        # Non-Trainable params for eval stage
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has dimension: [Batch_Size, Channels, Height, Width]
        
        # New size -> [1, Channels. 1, 1] for broadcasting
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        
        if self.training:
            # The result will be a vector of dimension [Channels]
            mean = x.mean(dim=(0, 2, 3)) # Mean on axes: 0 (Batch), 2 (Height), 3 (Width)
            var = x.var(dim=(0, 2, 3), unbiased=False) # Var on aexes: 0 (Batch), 2 (Height), 3 (Width)
            
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # At the inference stage, use use what we learned in the training phase
            mean = self.running_mean
            var = self.running_var
        
        # New size -> [1, Channels. 1, 1] for broadcasting
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        
        norm = (x - mean) / torch.sqrt(var + self.eps)
        scale = norm * weight + bias # Scaling (Gamma) + Translation (Beta)
        
        return scale

class ResidualBlock(nn.Module):
    def __init__(self, num_features: int, leaky: bool = False): # ResNet-14 uses for the first blocks num_features=64 hence the convolutional layers have a in_channels / out_channels = 64 and for the latter ones num_features=2*64=128, hence in_channels / out_channels = 128
        super().__init__()
        self.leaky = leaky
        self.norm1 = BatchNorm(num_features)
        self.norm2 = BatchNorm(num_features)
        # The recipe to kernel dimension unchanging is: (padding = kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x # Copy of input
        if self.leaky:
            x = LeakyReLU(self.conv1(self.norm1(x)))
        else:
            x = ReLU(self.conv1(self.norm1(x)))
        x = self.conv2(self.norm2(x)) # The final convolutional is liniar before the addition, hence we dont apply ReLU here
        skip_connection = x + identity
        if self.leaky:
            return LeakyReLU(skip_connection)
        else:
            return ReLU(skip_connection)
    
class ResNet_12(nn.Module):
    """ResNet_12 model architecture in pure PyTorch.
    
    It contains 14 layers in total: 1 initial convolutional layer, 1 initial max pooling layer,
    followed by 5 residual blocks (each containing 2 convolutional layers, totaling 10 conv layers),
    1 global average pooling layer (part of the latter layers, as was the standard back then), and 1 final fully connected layer.
    The network is optimized for asymmetric 2D sequential data or spectrograms rather than standard square images.
    
    Layer Breakdown:
    
    1. Input: 1x7x96 feature matrix w/ 1 channel (e.g, mono audio spectrogram)
    2. C1 (Convolution): 3x3 filters, 64 feature maps, stride 2, pad 1, output size 64x4x48
    3. S2 (MaxPool): 3x3 window, stride 2, pad 1, output size 64x2x24 # Reduces spatial dimensions early to minimize downstream computation
    4. ResNet Block-1 (C3, C4): Two 3x3 conv layers, 64 feature maps, stride 1, pad 1, output size 64x2x24 w/ skip connections
    5. ResNet Block-2 (C5, C6): Two 3x3 conv layers, 64 feature maps, stride 1, pad 1, output size 64x2x24 w/ skip connections
    6. ResNet Block-3 (C7, C8): Two 3x3 conv layers, 64 feature maps, stride 1, pad 1, output size 64x2x24 w/ skip connections
    7. ResNet Block-4 (C9, C10): Two 3x3 conv layers, 64 feature maps, stride 1, pad 1, output size 64x2x24 w/ skip connections
    8. ResNet Block-5 (C11, C12): Two 3x3 conv layers, 64 feature maps, stride 1, pad 1, output size 64x2x24 w/ skip connections
    9. GAP (Global Average Pooling): 2x24 kernel size, output size 64x1x1 # Collapses all spatial elements per channel into a single mean value
    10. F13 (Fully Connected Layer): 96 output neurons (will implement nn.Flatten -> 64 connected to 96 target classes)
    
    Notes taken while writing this Layer Breakdown:
    - In contrast to standard few-shot ResNet-12 architectures that increase feature map depth (e.g, 64 -> 160 -> 320 -> 640), this custom variant maintains a constant depth of 64 channels across all 5 blocks, which keeps the total parameter footprint exceptionally lightweight.
    - Because the spatial dimensions become highly compressed (2x24) right after the initial MaxPool layer, the 5 consecutive ResNet blocks use padding=1 and stride=1. This geometric trick acts to preserve the remaining structural matrix completely intact, allowing deep information extraction w/o losing coordinates before the final pool.
    - Stacking 10 convolutional layers inside the residual blocks allows the network to learn intricate hierarchical transformations. Since each block bypasses its original input via a shortcut line, gradients can flow backwards unimpeded during training, preventing the vanishing gradient problem
    - GAP behaves as a robust spatial regularizer. By averaging out the entire 2x24 feature plane down to 1x1, it makes the network invariant to translation shifts in the input matrix and havily discourages overfitting compared to flattening a whole matrix directly into an expensive linear layer.
    """
    def __init__(self):
        super().__init__()
        
        self.c1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.r1 = ResidualBlock(num_features=64)
        self.r2 = ResidualBlock(num_features=64)
        self.r3 = ResidualBlock(num_features=64)
        self.r4 = ResidualBlock(num_features=64)
        self.r5 = ResidualBlock(num_features=64)
        
        self.avgpool2 = nn.AvgPool2d(kernel_size=(2, 24)) # Alternative: nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc7 = nn.Linear(in_features=64*2*24, out_featues=96) # in_features[C, H, W] = 2072
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry input size: [Batch, 1, 7, 96]
        x = ReLU(self.c1(x))
        x = self.maxpool1(x)
        
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.r5(x)
        
        x = self.avgpool2(x)
        
        # start_dim=1 assures that we flatten just [C, H, W], w/o the batch
        x = torch.flatten(x, start_dim=1)
        
        x = self.fc7(x)
        
        return x
    
    """Returns the total number of parameters of ResNet_12"""
    def params(self):
        return sum(p.numel() for p in self.parameters())
    
    """Training + Evaluation Metrics"""
    def fit(self, train_loader, val_loader, device=None, track=None, epochs=EPOCHS):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
            
        self.to(device)
        
        crit = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        
        train_loss_history = []
        val_loss_history = []
        
        y_true_epoch = None
        y_pred_epoch = None
        
        print(f"ResNet_12 training will start on: {device.type.upper()}")
        print("=" * 60)
        
        for epoch in range(epochs):
            self.train()
            running_loss = 0.
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad                
                outputs = self(inputs)
                
                loss = crit(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                if batch_idx % 100 == 0:
                    print(f"Epoch: {epoch+1} | Batch: {batch_idx:03d} | Batch Loss {loss.item():.4f}")
                    
            epoch_loss = running_loss / len(train_loader)
            
            is_last_epoch = (epoch == epochs - 1)
            should_return_arrays = is_last_epoch and track
            val_accuracy, val_loss, y_true_epoch, y_pred_epoch = eval(val_loader, device=device, return_arrays=should_return_arrays)
            
            train_loss_history.append(epoch_loss)
            val_loss_history.append(val_loss)
            
            print(f"Epoch {epoch+1:02d}/{epochs:02d} completed | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            
        if track:
            return train_loss_history, val_loss_history, y_true_epoch, y_pred_epoch
        
        return train_loss_history, val_loss_history, None, None
            
            
    
    def eval(self, val_loader, device=None, verbose=False, return_arrays=False):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        self.to(device) # Moves the model weights to device
        self.eval()
        
        correct = 0
        total = 0
        running_val_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        # Initializing collections to hold target targets and model outputs
        y_true = [] if return_arrays else None
        y_pred = [] if return_arrays else None
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if return_arrays:
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(predicted.cpu().tolist())
                    
            accuracy = 100 * correct / total
            avg_val_loss = running_val_loss / len(val_loader)
            
            if verbose:
                print(f"Total samples evaluated: {total}")
                print(f"Correct predictions: {correct}")
            
            return accuracy, avg_val_loss, y_true, y_pred
    
    DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alexnet_model.pth')
    
    """Load model class method"""
    def load(self, path=None, device=None):
        """Loads the model's weights from a file."""
        path = path or self.DEFAULT_WEIGHTS
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
            
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)
        self.to(device)
        
        print(f"ResNet_12 model loaded from {path} to {device}")
    
    """Save model class method"""
    def save(self, path=None):
        """Saves the model's weights to a file."""
        path = path or self.DEFAULT_WEIGHTS
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        torch.save(self.state_dict(), path)
        print(f"ResNet_12 model saved to {path}")