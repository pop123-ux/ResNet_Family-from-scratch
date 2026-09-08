import time
import os
import torch
import torch.nn as nn
from src import ReLU, LeakyReLU

EPOCHS: int = 30

class BatchNorm(nn.Module):
    # num_features is of size C (Channels)
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def fit(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W]
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        
        if self.training:
            mean = x.mean(dim=(0, 2, 3)) # Mean on axes: [B, H, W]
            var = x.var(dim=(0, 2, 3), unbiased=False) # Var on axes: [B, H, W]
            
            with torch.no_grad():
                self.running_mean = (1 - self.mometum) * self.running_mean + self.mometum * mean
                self.running_var = (1 - self.mometum) * self.running_mean + self.mometum * var
                
        else:
            # Use the variables that don't update gradients
            mean = self.running_mean
            var = self.running_var 
            
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        
        norm = (x - mean) / torch.sqrt(var + self.eps)
        scale = norm * weight + bias
        
        return scale
    
class ResidualBlock(nn.Module):
    """
    Custom ResidualBlock solely for this ResNet_50-like architecture w/ Blocks having pattern: [1x1, 3x3, 1x1]
    """
    def __init__(self, in_features: int, base_features: int, stride: int = 1, leaky: bool = False):
        super().__init__()
        self.leaky = leaky
        self.activation = LeakyReLU(0.1) if leaky else ReLU()
        
        # Final dimension is always 4x base_features
        out_features = 4 * base_features
        
        # 1x1 Compression layer
        self.conv1 = nn.Conv2d(in_features, base_features, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(base_features)
        
        # 3x3 Spatial Convolution layer (Applies downsampling via stride when transitioning groups)
        self.conv2 = nn.Conv2d(base_features, base_features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm(base_features)
        
        # 1x1 Expansion layer
        self.conv3 = nn.Conv2d(base_features, out_features, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(out_features)
        
        # Adaptive Skip-Connection (adjusts shape if spatial dimension drops)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False),
                BatchNorm(out_features)
            )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        x = self.activation(self.bn1(self.conv1))
        x = self.activation(self.bn2(self.conv2))
        x = self.bn3(self.conv3)
        
        return self.activation(x + identity)
    
class ResNet_50(nn.Module):
    """ResNet_50 model architecture in pure PyTorch.
    
    It contains 50 layers in total: 1 initial convolutional (stem) layer, 1 initial max pooling layern
    followed by 4 residual blocks (containing different shapes and layer sizes, totaling 48 conv layers),
    1 global average pooling layer, and 1 final fully connected layer.
    The network is optimized for square 2D matrices (As ResNet_18 & Resnet_34 implementations), adapted to process intermediate spatial scaled w/o overly aggressive early downsampling.
    
    Layer Breakdown:
    
    1. Input: 3x224x224 feature matrix w/ 3 channels (e.g, standard RGB image)
    2. C1 (Convolution): 7x7 filters, 64 feature maps, stride 2, pad 3
    3. S2 (MaxPool): 3x3 window, stride 2, pad 1
    4. ResNet Layer-1 (C3-C11): Nine conv layers
    5. ResNet Layer-2 (C12-C23): Twelve conv layers
    6. ResNet Layer-3 (C24-C41): Eighteen conv layers
    7. ResNet Layer-4 (C42-C50): Nine conv layers
    8. GAP (Global Average Pooling): Here implemented as the modern Adaptive Pooling Layer, collapses all spatial elements per channel into a single mean value
    9. F50 (Fully Connected Layer): Custom output neurons (will implement torch.flatten in the forward pass -> 2048 connected to target classification labels)
    
    Notes taken while writing this Layer Breakdown:
    - In contrast to other ResNet architectures, this scaled version I could say, implements a more complex residual layer (having 3 convolutional layers each) with different kernel sizes, extracting more complex spatial information at the expense of compute
    - This architecture introduces a 3-layer "bottleneck" design per residual block (using 1x1, 3x3, and 1x1 convolutions). The initial 1x1 convolution reduces dimensionality, the 3x3 convolution operated on a smaller channel volume, and the final 1x1 convolution restores the high-dimensional projection, significantly limiting parameter explosion while deepening the model. 
    - Definetely not an easy piece to train at this layer scale :)
    """
    DEFAULT_WEIGHTS = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ResNet50_model.pth')
    if '__file__' in locals()
    else 'ResNet50_model.pth'
    )
    def __init__(self, in_channels: int = 3, num_classes: int = 1000, leaky: bool = False):
        super().__init__()
        self.activation = LeakyReLU() if leaky else ReLU()
        
        # Initial Stem Layer (Input 224x224 -> 112x112)
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = BatchNorm(64)
        
        # Max Pooling (112x112 -> 56x56)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # --- Each sublayer will be the same 3-layer sequence: Conv2d_1(1x1) -> BatchNorm1 -> Conv2d_2(3x3) -> BatchNorm2 -> Conv2d_3(1x1) -> BatchNorm3 ---
        
        # ResNet Layer-1 - Output Shape: 56x56x256
        # 3 sublayers x 3 = 9 total convolutions
        self.layer1_1 = ResidualBlock(in_features=64, base_features=64, stride=1, leaky=leaky)
        self.layer1_2 = ResidualBlock(in_features=256, base_features=64, stride=1, leaky=leaky)
        self.layer1_3 = ResidualBlock(in_features=256, base_features=256, stride=1, leaky=leaky)
        
        # ResNet Layer-2 - Output Shape: 28x28x512
        # 4 sublayers x 3 = 12 total convolutions
        self.layer2_1 = ResidualBlock(in_features=256, base_features=128, stride=2, leaky=leaky)
        self.layer2_2 = ResidualBlock(in_features=512, base_features=128, stride=1, leaky=leaky)
        self.layer2_3 = ResidualBlock(in_features=512, base_features=128, stride=1, leaky=leaky)
        self.layer2_4 = ResidualBlock(in_features=512, base_features=128, stride=1, leaky=leaky)
        
        # ResNet Layer-3 - Output Shape: 14x14x1024
        # 6 sublayers x 3 = 18 total convolutions
        self.layer3_1 = ResidualBlock(in_features=512, base_features=256, stride=2, leaky=leaky)
        self.layer3_2 = ResidualBlock(in_features=1024, base_features=256, stride=1, leaky=leaky)
        self.layer3_3 = ResidualBlock(in_features=1024, base_features=256, stride=1, leaky=leaky)
        self.layer3_4 = ResidualBlock(in_features=1024, base_features=256, stride=1, leaky=leaky)
        self.layer3_5 = ResidualBlock(in_features=1024, base_features=256, stride=1, leaky=leaky)
        self.layer3_6 = ResidualBlock(in_features=1024, base_features=256, stride=1, leaky=leaky)
        
        # ResNet Layer-4 - Output Shape: 7x7x2048
        # 3 sublayers x 3 = 9 total convolutions
        self.layer4_1 = ResidualBlock(in_features=1024, base_features=512, stride=2, leaky=leaky)
        self.layer4_2 = ResidualBlock(in_features=2048, base_features=512, stride=1, leaky=leaky)
        self.layer4_3 = ResidualBlock(in_features=2048, base_features=512, stride=1, leaky=leaky)
        
        # GAP (Global Average Pooling) - Computes the global statistic of the features and condenses it in a vector - 7x7 spatial size -> 1x1 vector
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc50 = nn.Linear(2048*1*1, num_classes)
    
    def fit(self, x: torch.Tensor) -> torch.Tensor:
        
        # Stem Execution
        x = self.activation(self.norm1(self.c1))
        x = self.maxpool1(x) # stride 2
        
        # Layer 1: 3 total residual blocks w/ 3 layers each
        # ---
        # Resolution reduction via stride isn't applied in the first layer, since maxpool1 already reduces it to 56x56, its primary task is expanding the channel count from 64 to 256
        
        x = self.layer1_1(x) # Input: [B, 64, 56, 56] -> [B, 256, 56, 56]
        x = self.layer1_2(x) # Input: [B, 256, 56, 56] -> [B, 256, 56, 56]
        x = self.layer1_3(x) # Input: [B, 256, 56, 56] -> [B, 256, 56, 56]
        
        # Layer 2 - 4 total residual blocks w/ 3 layers each
        # ---
        # This layer reduces spatial resolution from 56 to 28 using a stride of 2 in the first block, and doubles the feature map depth from 256 to 512 channels
        
        x = self.layer2_1(x) # Input: [B, 256, 56, 56] -> [B, 512, 28, 28] | stride 2
        x = self.layer2_2(x) # Input: [B, 512, 28, 28] -> [B, 512, 28, 28]
        x = self.layer2_3(x) # Input: [B, 512, 28, 28] -> [B, 512, 28, 28]
        x = self.layer2_4(x) # Input: [B, 512, 28, 28] -> [B, 512, 28, 28]
        
        # Layer 3 - 6 total residual blocks w/ 3 layers each
        # ---
        # The longest block sequence in ResNet-50, halving resolution from 28 to 14 via stride in the first block and expands channels from 512 to 1024
        
        x = self.layer3_1(x) # Input: [B, 512, 28, 28] -> [B, 1024, 14, 14] | stride 2
        x = self.layer3_2(x) # Input: [B, 1024, 14, 14] -> [B, 1024, 14, 14]
        x = self.layer3_3(x) # Input: [B, 1024, 14, 14] -> [B, 1024, 14, 14]
        x = self.layer3_4(x) # Input: [B, 1024, 14, 14] -> [B, 1024, 14, 14]
        x = self.layer3_5(x) # Input: [B, 1024, 14, 14] -> [B, 1024, 14, 14]
        x = self.layer3_6(x) # Input: [B, 1024, 14, 14] -> [B, 1024, 14, 14]
        
        # Layer 4 - 3 total residual blocks w/ 3 layers each
        # ---
        # The final convolutional feature grouping, dropping spatial resolution from 14 to 7 and maximizes depth representation from 1024 to 2048 channels
        
        x = self.layer4_1(x) # Input: [B, 1024, 14, 14] -> [B, 2048, 7, 7] | stride 2
        x = self.layer4_2(x) # Input: [B, 2048, 7, 7] -> [B, 2048, 7, 7]
        x = self.layer4_3(x) # Input: [B, 2048, 7, 7] -> [B, 2048, 7, 7]
        
        # Output: Pool, Flatten, Classify
        x = self.avgpool2(x) # Input: [B, 2048, 7, 7] -> [B, 2048, 1, 1]
        x = torch.flatten(x, start_dim=1) # Input: [B, 2048, 1, 1] -> [B, 2048] | In [B, C, H, W], C is the dim on pos 1
        x = self.fc50(x) # Input: [B, 2048] -> [B, num_classes]
        
    """Returns the total number of parameters of ResNet_50"""
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
        
        print(f"ResNet_50 training will start on: {device.type.upper()}")
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
        
        print(f"ResNet_50 model loaded from {path} to {device}")
    
    """Save model class method"""
    def save(self, path=None):
        """Saves the model's weights to a file."""
        path = path or self.DEFAULT_WEIGHTS
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        torch.save(self.state_dict(), path)
        print(f"ResNet_50 model saved to {path}")
