import time
import os
import torch
import torch.nn as nn
from src.utils import ReLU, LeakyReLU

EPOCHS: int = 30

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.ones(num_features))
        
        self.eps = eps
        self.momentum = momentum
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has dim: [B, C, H, W]
        weight = self.weight.view(1,-1,1,1)
        bias = self.bias.view(1,-1,1,1)
        
        if self.training:
            mean = x.mean(dim=(0, 2, 3)) # Mean on axes: [B, H, W]
            # PyTorch uses unbiased=False implicitly for BatchNorm in running stats
            var = x.var(dim=(0, 2, 3), unbiased=False) # Var on axes: [B, H, W]
            
            with torch.no_grad():
                # Exponential Moving Average - The Network memorizes the data global statistic in order to use it in the inference stage
                self.running_mean = (1 - self.momentum) * self.running_mean  + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                
        else:
            mean = self.running_mean
            var = self.running_var
            
        # [1, C, 1, 1] for broadcasting
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        
        norm = (x - mean) / torch.sqrt(var + self.eps)
        scale = norm * weight + bias
        
        return scale
    
class ResidualBlock(nn.Module):
    """A flexible residual block that can group n conv layers together 
    and dynamically handle downsampling/projection shortcuts.
    """
    def __init__(self, in_features: int, out_features: int, stride: int = 1, num_layers: int = 2, leaky: bool = False): # ResNet-18 uses num_features = 64, as the in_channels = out_channels of the convolutional layers stacked inside the residual block 
        super().__init__()
        self.leaky = leaky
        self.activation = LeakyReLU(0.1) if leaky else ReLU()
        
        layers = []
        current_in = in_features
        
        # Build the sequential convolutional path inside the block
        for i in range(num_layers):
            # Stride is only applied to the first layer of the block
            layer_stride = stride if i == 0 else 1
            layers.append(nn.Conv2d(current_in, out_features, kernel_size=3, stride=layer_stride, padding=1, bias=False))
            layers.append(BatchNorm(out_features))
            if i < num_layers - 1:
                layers.append(self.activation)
            current_in = out_features
        
        self.conv_path = nn.Sequential(*layers)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False),
                BatchNorm(out_features)
            )        
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.conv_path(x)
        return self.activation((x + identity))
    
class ResNet_18(nn.Module):
    """ResNet_18 model architecture in pure PyTorch.
    
    It contains 19 layers in total: 1 initial convolutional layer, 1 initial max pooling layer,
    followed by 4 residual blocks (each containing 4 convolutional layers organized in 2 sub-blocks, totaling 16 conv layers),
    1 global average pooling layer, and 1 final fully connected layer.
    The network is optimized for square 2D matrices, adapted to process intermediate spatial scaled w/o overly aggressive early downsampling.
    
    Layer Breakdown:
    
    1. Input: 3x100x100 feature matrix w/ 3 channels (e.g., standard RGB image)
    2. C1 (Convolution): 7x7 filters, 64 feature maps, stride 1, pad 3, output size 64x100x100
    3. S2 (MaxPool): 3x3 window, stride 1, pad 1, output size 64x100x100 # Maintains high spatial resolution early to preserve fine-grained structural details
    4. ResNet Layer-1 (C3, C4, C5, C6): Four 3x3 conv layers, 64 feature maps, stride 1, pad 1, output size 64x100x100 w/ identity skip connections
    5. ResNet Layer-2 (C7, C8, C9, C10): Four 3x3 conv layers, 128 feature maps, transition stride 2, pad 1, output size 128x50x50 w/ projection skip connections
    6. ResNet Layer-3 (C11, C12, C13, C14): Four 3x3 conv layers, 256 feature maps, transition stride 2, pad 1, output size 256x25x25 w/ projection skip connections
    7. ResNet Layer-4 (C15, C16, C17, C18): Four conv layers, 512 feature maps, transition block uses a customized 5x5 filter w/ stride 5 and pad 0, output size 512x5x5 w/ projection skip connections
    8. GAP (Global Average Pooling): 5x5 kernel size, output size 512x1x1 # Collapses all spatial elements per channel into a single mean value
    9. F10 (Fully Connected Layer): Custom output neurons (will implement nn.Flatten -> 512 connected to target classification classes)
    
    Notes taken while writing this Layer Breakdown:
    - In contrast to other ResNet architectures that use aggressive stride=2 in both the initial convolution and the MaxPool layers (which would immediately crush a 100x100 input down to 25x25), this custom variant maintains a full 100x100 spatial resolution through Layer-1 to allow deeper representation learning on smaller input sizes.
    - Because a standard 3x3 convolution with stride=2 applied to a 25x25 matrix results in a non-integer fraction rounded down to 12x12 (due to PyTorch's floor operation), the transition block in Layer-4 is explicitly designed w/ a 5x5 kernel, stride=5, and padding=0. This geometric configuration perfectly scales the spatial grid down from 25x25 to an exact 5x5 output.
    - GAP behaves as a robust spatial regularizer. By averaging out the final 5x5 feature plane down to 1x1, it enforces translation invariance and heavily reduces the parameter footprint of the classifier head, drastically minimizing overfitting compared to flattening a 5x5x512 matrix directly into a massive linear layer.
    - Extra: When a BatchNorm layer immediately follows a convolutional layer, the convolutional bias parameter b becomes completely redundant. Adding a static bias b simply shifts the distribution's mean by that exact amount b. When BatchNorm computes the new mean and subtracts it, the bias b cancels out perfectly. Setting bias=False ensures the model does not waste VRAM or training time updating useless parameters
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, leaky: bool = False):
        super().__init__()
        self.leaky = leaky
        self.activation = LeakyReLU() if leaky else ReLU()
        
        # Initial stem layer with stride=2 to downsample input from 100x100 to 50x50
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = BatchNorm(64)
        
        # Initial Max Pooling (3x3) drops resolution from 50x50 to 25x25
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 1 - Resolution [25, 25]
        # 4 identical 3x3 conv layers split into 2 blocks of 2 layers each
        self.layer1_block1 = ResidualBlock(64, 64, stride=1, num_layers=2, leaky=self.leaky)
        self.layer1_block2 = ResidualBlock(64, 64, stride=1, num_layers=2, leaky=self.leaky)
        
        # Layer 2 - Resolution [13, 13]
        # 4 identical 3x3 conv layers split into 2 blocks of 2 layers each (First block downsamples)
        self.layer2_downsample = ResidualBlock(64, 128, stride=2, num_layers=2, leaky=self.leaky)
        self.layer2_identical = ResidualBlock(128, 128, stride=1, num_layers=2, leaky=self.leaky)
        
        # Layer 3 - Shape [7, 7]
        # 4 identical 3x3 conv layers split into 2 blocks of 2 layers each (First block downsamples)
        self.layer3_downsample = ResidualBlock(128, 256, stride=2, num_layers=2, leaky=self.leaky)
        self.layer3_identical = ResidualBlock(256, 256, stride=1, num_layers=2, leaky=self.leaky)
        
        # Layer 4 - Shape [4, 4]
        # 4 identical 3x3 conv layers split into 2 blocks of 2 layers each (First block downsamples)
        self.layer4_downsample = ResidualBlock(256, 512, stride=2, num_layers=2, leaky=self.leaky)
        self.layer4_identical = ResidualBlock(512, 512, stride=1, num_layers=2, leaky=self.leaky)
        
        # Global Average Pooling (4x4 spatial size adaptive reduction -> 1x1)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc10 = nn.Linear(in_features=512*1*1, out_features=num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial layers
        x = self.activation((self.norm1(self.c1(x))))
        x = self.maxpool1(x)
        
        # Layer 1
        x = self.layer1_block1(x)
        x = self.layer1_block2(x)
        
        # Layer 2
        x = self.layer2_downsample(x)
        x = self.layer2_identical(x)
        
        # Layer 3
        x = self.layer3_downsample(x)
        x = self.layer3_identical(x)
        
        # Layer 4
        x = self.layer4_downsample(x)
        x = self.layer4_identical(x)
        
        # Pool, Flatten, and Classify
        x = self.avgpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc10(x)
        
        return x
    
    """Returns the total number of parameters of ResNet_18"""
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
        # Learning rate downscaled 10x at epochs 30, 60, 90
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
        train_loss_history = []
        val_loss_history = []
        
        y_true_epoch = None
        y_pred_epoch = None
        
        print(f"ResNet_18 training will start on: {device.type.upper()}")
        print("=" * 60)
        
        for epoch in range(epochs):
            self.train()
            running_loss = 0.
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()                
                outputs = self(inputs)
                
                loss = crit(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                current_lr = scheduler.get_last_lr()[0]
                if batch_idx % 100 == 0:
                    print(f"Epoch: {epoch+1} | Lr: {current_lr} | Batch: {batch_idx:03d} | Batch Loss {loss.item():.4f}")
                    
            epoch_loss = running_loss / len(train_loader)
            
            is_last_epoch = (epoch == epochs - 1)
            should_return_arrays = is_last_epoch and track
            val_accuracy, val_loss, y_true_epoch, y_pred_epoch = self.evaluate(val_loader, device=device, return_arrays=should_return_arrays)
            
            scheduler.step(val_loss)
            
            train_loss_history.append(epoch_loss)
            val_loss_history.append(val_loss)
            
            print(f"Epoch {epoch+1:02d}/{epochs:02d} completed | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            
        if track:
            return train_loss_history, val_loss_history, y_true_epoch, y_pred_epoch
        
        return train_loss_history, val_loss_history, None, None
            
            
    
    def evaluate(self, val_loader, device=None, verbose=False, return_arrays=False):
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
    
    DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ResNet18_model.pth')
    
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
        
        print(f"ResNet_18 model loaded from {path} to {device}")
    
    """Save model class method"""
    def save(self, path=None):
        """Saves the model's weights to a file."""
        path = path or self.DEFAULT_WEIGHTS
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        torch.save(self.state_dict(), path)
        print(f"ResNet_18 model saved to {path}")