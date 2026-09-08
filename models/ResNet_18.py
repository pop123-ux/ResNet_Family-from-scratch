import time
import os
import torch
import torch.nn as nn
from src import ReLU, LeakyReLU

EPOCHS: int = 30

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features)) # gamma γ
        self.bias = nn.Parameter(torch.zeros(num_features)) # beta β
        
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
            # Current-batch normalization uses the biased variance estimate (unbiased=False);
            # running_var is updated below using the unbiased variance estimate
            var = x.var(dim=(0, 2, 3), unbiased=False) # Var on axes: [B, H, W]
            
            with torch.no_grad():
                running_var_batch = x.var(
                dim=(0, 2, 3),
                unbiased=True,
                )
                
                # Exponential Moving Average - The Network memorizes the data global statistic in order to use it in the inference stage
                self.running_mean = (1 - self.momentum) * self.running_mean  + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * running_var_batch
                
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

    It contains 18 trainable layers in total: 1 initial convolutional layer,
    followed by 4 residual stages containing 8 residual blocks with 2 convolutional layers each (16 conv layers total),
    and 1 final fully connected classification layer.
    The network is optimized for square 100x100 RGB inputs and deliberately delays spatial downsampling compared to the canonical ImageNet ResNet-18.

    Layer Breakdown:

    1. Input: 3x100x100 feature matrix w/ 3 channels (e.g., standard RGB image)
    2. C1 (Convolution): 7x7 filters, 64 feature maps, stride 1, pad 3, output size 64x100x100
    3. S2 (MaxPool): 3x3 window, stride 1, pad 1, output size 64x100x100 # Preserves the full spatial resolution through the stem
    4. ResNet Layer-1 (C2-C5): Two residual blocks x 2 conv layers each, 64 feature maps, stride 1, output size 64x100x100 w/ identity skip connections
    5. ResNet Layer-2 (C6-C9): Two residual blocks x 2 conv layers each, 128 feature maps, transition stride 2 downsamples to 50x50, output size 128x50x50 w/ projection skip connection in the first block
    6. ResNet Layer-3 (C10-C13): Two residual blocks x 2 conv layers each, 256 feature maps, transition stride 2 downsamples to 25x25, output size 256x25x25 w/ projection skip connection in the first block
    7. ResNet Layer-4 (C14-C17): Two residual blocks x 2 conv layers each, 512 feature maps, transition stride 5 downsamples to 5x5, output size 512x5x5 w/ projection skip connection in the first block
    8. GAP (Global Average Pooling): 5x5 average pooling reduces each 512-channel feature map from 5x5 to 1x1
    9. F18 (Fully Connected Layer): torch.flatten transforms 512x1x1 -> 512 features, then nn.Linear maps 512 features to num_classes

    Notes taken while writing this Layer Breakdown:
    - This implementation keeps the standard ResNet-18 basic-block structure of 2 convolutional layers per residual block and 2 blocks per stage, but deviates from the canonical architecture by using stride 1 in both the initial convolution and max-pooling layer.
    - Spatial resolution is therefore preserved at 100x100 through the stem and reduced later through the residual stages: 100x100 -> 50x50 -> 25x25 -> 5x5.
    - Projection shortcuts use a 1x1 convolution whenever either the spatial resolution or channel count changes, ensuring the identity tensor matches the main path before residual addition.
    - GAP collapses the final 5x5 spatial feature maps into one value per channel, reducing the classifier input to only 512 features instead of flattening the complete 512x5x5 representation.
    - Extra: When BatchNorm immediately follows a convolutional layer, the convolutional bias becomes redundant because BatchNorm subtracts the channel mean and then applies its own learnable beta shift. Setting bias=False therefore avoids unnecessary parameters.
    """
    
    DEFAULT_WEIGHTS = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ResNet18_model.pth')
    if '__file__' in locals()
    else 'ResNet18_model.pth'
    )
    
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, leaky: bool = False):
        super().__init__()
        self.leaky = leaky
        self.activation = LeakyReLU() if leaky else ReLU()
        
        # Initial stem layer
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False)
        self.norm1 = BatchNorm(64)
        
        # Initial Max Pooling (3x3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # Layer 1 - Resolution [100, 100]
        # 4 identical 3x3 conv layers split into 2 blocks of 2 layers each
        self.layer1_block1 = ResidualBlock(64, 64, stride=1, num_layers=2, leaky=self.leaky)
        self.layer1_block2 = ResidualBlock(64, 64, stride=1, num_layers=2, leaky=self.leaky)
        
        # Layer 2 - Resolution [50, 50] (stride=2 halves 100x100)
        # 1 downsampling residual block followed by 1 identity residual block
        self.layer2_downsample = ResidualBlock(64, 128, stride=2, num_layers=2, leaky=self.leaky)
        self.layer2_identical = ResidualBlock(128, 128, stride=1, num_layers=2, leaky=self.leaky)
        
        # Layer 3 - Shape [25, 25] (stride=2 halves 50x50)
        # 1 downsampling residual block followed by 1 identity residual block
        self.layer3_downsample = ResidualBlock(128, 256, stride=2, num_layers=2, leaky=self.leaky)
        self.layer3_identical = ResidualBlock(256, 256, stride=1, num_layers=2, leaky=self.leaky)
        
        # Layer 4 - Shape [5, 5] (stride=5 reduces 25x25 down to 5x5)
        # 1 downsampling residual block followed by 1 identity residual block
        self.layer4_downsample = ResidualBlock(256, 512, stride=5, num_layers=2, leaky=self.leaky)
        self.layer4_identical = ResidualBlock(512, 512, stride=1, num_layers=2, leaky=self.leaky)
        
        # For the intended 100x100 input, the final 5x5 feature map is globally averaged to 1x1
        self.avgpool2 = nn.AvgPool2d(kernel_size=5)
        
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
        use_amp = device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        # Learning rate dynamic 10x downscaling
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//3, 2*epochs//3, 5*epochs//6], gamma=0.1)
        train_loss_history = []
        val_loss_history = []
        
        y_true_epoch = None
        y_pred_epoch = None
        
        print(f"ResNet_18 training will start on: {device.type.upper()}")
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
                    print(f"Epoch: {epoch+1} | Lr: {current_lr} | Batch: {batch_idx:03d} | Batch Loss {loss.item():.4f}")
                    
            epoch_loss = running_loss / len(train_loader)
            
            is_last_epoch = (epoch == epochs - 1)
            should_return_arrays = is_last_epoch and track
            val_accuracy, val_loss, y_true_epoch, y_pred_epoch = self.evaluate(val_loader, device=device, return_arrays=should_return_arrays)
            
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