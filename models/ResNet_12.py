"""Custom lightweight ResNet variant for asymmetric 2D inputs."""
import time
import os
import torch
import torch.nn as nn
from src import ReLU, LeakyReLU

EPOCHS: int = 30

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, device=None):
        super().__init__()
        # Trainable params
        self.weight = nn.Parameter(torch.ones(num_features)) # gamma γ
        self.bias = nn.Parameter(torch.zeros(num_features)) # beta β
        
        self.eps = eps
        self.momentum = momentum
        
        # Non-trainable running statistics, updated during training and used during evaluation
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
            var = x.var(dim=(0, 2, 3), unbiased=False) # Variance over axes: 0 (Batch), 2 (Height), 3 (Width)
            
            with torch.no_grad():
                running_var_batch = x.var(
                dim=(0, 2, 3),
                unbiased=True,
                )
                
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * running_var_batch
        else:
            # During evaluation, use the running statistics accumulated during training
            mean = self.running_mean
            var = self.running_var
        
        # New size -> [1, Channels. 1, 1] for broadcasting
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        
        norm = (x - mean) / torch.sqrt(var + self.eps)
        scale = norm * weight + bias # Scaling (Gamma) + Translation (Beta)
        
        return scale

class ResidualBlock(nn.Module):
    def __init__(self, num_features: int, kernel_size: int = 3, padding: int = 1, leaky: bool = False): # ResNet-12 uses num_features = 64, as the in_channels = out_channels of the convolutional layers stacked inside the residual block 
        super().__init__()
        self.leaky = leaky
        self.activation = LeakyReLU() if leaky else ReLU()
        
        self.norm1 = BatchNorm(num_features)
        # Since BatchNorm includes learnable parameters, we need to instantiate another instance attribute which is going to be used for the input to the second convolutional layer
        self.norm2 = BatchNorm(num_features)
        
        self.conv1 = nn.Conv2d(in_channels=num_features, out_channels=num_features, stride=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels=num_features, out_channels=num_features, stride=1, kernel_size=kernel_size, padding=padding, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x # Copy of input
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        # The final convolutional block remains linear right before addition
        skip_connection = x + identity
        return self.activation(skip_connection)
    
class ResNet_12(nn.Module):
    """ResNet_12 model architecture in pure PyTorch.
    
    It contains 12 trainable layers in total: 1 initial convolutional layer,
    followed by 5 residual blocks containing 2 convolutional layers each (10 conv layers total),
    and 1 final fully connected classification layer.
    
    Max pooling and adaptive global average pooling are used as non-parametric operations.
    
    Layer Breakdown:
    
    1. Input: 1x7x96 feature matrix w/ 1 channel (e.g, mono audio spectrogram)
    2. C1 (Convolution): 3x3 filters, 64 feature maps, stride 2, pad 1, output size 64x4x48
    3. S2 (MaxPool): 3x3 window, stride 2, pad 1, output size 64x2x24 # Reduces spatial dimensions early to minimize downstream computation
    4. ResNet Block-1 (C2-3): Two 3x3 conv layers, 64 feature maps, stride 1, pad 1, output size 64x2x24 w/ skip connections
    5. ResNet Block-2 (C4-5): Two 3x3 conv layers, 64 feature maps, stride 1, pad 1, output size 64x2x24 w/ skip connections
    6. ResNet Block-3 (C6-7): Two 3x3 conv layers, 64 feature maps, stride 1, pad 1, output size 64x2x24 w/ skip connections
    7. ResNet Block-4 (C8-9): Two 3x3 conv layers, 64 feature maps, stride 1, pad 1, output size 64x2x24 w/ skip connections
    8. ResNet Block-5 (C10-11): Two 3x3 conv layers, 64 feature maps, stride 1, pad 1, output size 64x2x24 w/ skip connections
    9. GAP (Global Average Pooling): Adaptive average pooling reduces each 64-channel spatial feature map to 1x1.
    10. F12 (Fully Connected Layer): 96 output neurons
    
    Notes taken while writing this Layer Breakdown:
    - In contrast to standard few-shot ResNet-12 architectures that increase feature map depth (e.g, 64 -> 160 -> 320 -> 640), this custom variant maintains a constant depth of 64 channels across all 5 blocks, which keeps the total parameter footprint exceptionally lightweight.
    - Because the spatial dimensions become highly compressed (2x24) right after the initial MaxPool layer, the 5 consecutive ResNet blocks use padding=1 and stride=1. This geometric trick acts to preserve the remaining structural matrix completely intact, allowing deep information extraction w/o losing coordinates before the final pool.
    - Stacking 10 convolutional layers inside the residual blocks allows the network to learn intricate hierarchical transformations. Since each block bypasses its original input via a shortcut connection, skip connections provide shorter gradient paths, improving gradient flow and helping mitigate the vanishing-gradient problem.
    - GAP reduces sensitivity to exact spatial locations and greatly reduces the classifier parameter count, providing a useful regularizing effect.
    """
    
    DEFAULT_WEIGHTS = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ResNet12_model.pth')
    if '__file__' in locals()
    else 'ResNet12_model.pth'
    )
    
    def __init__(self, in_channels: int = 1, num_classes: int = 96, leaky: bool = False):
        super().__init__()
        self.leaky = leaky
        self.activation = LeakyReLU() if leaky else ReLU()
        
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = BatchNorm(64)
        
        # S2: drops spatial maps from [64, 4, 48] down to [64, 2, 24]
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 5 consecutive ResNet blocks maintaining a constant depth of 64 channels
        self.r1 = ResidualBlock(num_features=64, leaky=self.leaky)
        self.r2 = ResidualBlock(num_features=64, leaky=self.leaky)
        self.r3 = ResidualBlock(num_features=64, leaky=self.leaky)
        self.r4 = ResidualBlock(num_features=64, leaky=self.leaky)
        self.r5 = ResidualBlock(num_features=64, leaky=self.leaky)
        
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc7 = nn.Linear(in_features=64*1*1, out_features=num_classes) # 64x1x1 -> 64 input features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entry input size: [Batch, 1, 7, 96]
        x = self.activation(self.norm1(self.c1(x)))
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
        use_amp = device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        # Learning dynamic 10x downscaling
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//3, 2*epochs//3, 5*epochs//6], gamma=0.1)
        train_loss_history = []
        val_loss_history = []
        
        y_true_epoch = None
        y_pred_epoch = None
        
        print(f"ResNet_12 training will start on: {device.type.upper()}")
        print(f"Mixed precision: {'ENABLED' if use_amp else 'DISABLED'}")
        print("=" * 60)
        
        for epoch in range(epochs):
            self.train()
            running_loss = 0.
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    outputs = self(inputs)
                    loss = crit(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()

                current_lr = scheduler.get_last_lr()[0]
                if batch_idx % 100 == 0:
                    print(f"Epoch: {epoch+1} | Batch: {batch_idx:03d} | Batch Loss {loss.item():.4f}")
                    
            epoch_loss = running_loss / len(train_loader)
            
            is_last_epoch = (epoch == epochs - 1)
            should_return_arrays = is_last_epoch and track
            val_accuracy, val_loss, y_true_epoch, y_pred_epoch = self.evaluate(val_loader=val_loader, device=device, return_arrays=should_return_arrays)
            
            scheduler.step()
            
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
        use_amp = device.type == 'cuda'
        
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
                
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
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