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
        weight = self.weight.view(1, -1, 1, 1)
        bias =self.bias.view(1, -1, 1, 1)
        
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            
            with torch.no_grad():
                # EMA - The Network memorizes the data global statistic in order to use it in the inference stage
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        
        else:
            mean = self.running_mean
            var = self.running_var
            
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        
        norm = (x - mean) / torch.sqrt(var + self.eps)
        scale = norm * weight + bias
        
        return scale
    
class ResidualBlock(nn.Module):
    """A flexible residual block that can group n conv layers together
    and dynamically handle downsampling/projection shortcuts.
    """
    def __init__(self, in_features: int, out_features: int, stride: int = 1, num_layers: int = 2, leaky: bool = False):
        super().__init__()
        self.leaky = leaky
        self.activation = LeakyReLU(0.1) if leaky else ReLU()
        
        layers = []
        current_in = in_features
        
        # Build the sequential convolutional path inside the block
        for i in range(num_layers):
            # Only the first layer of the block uses the downsampling stride for the implementation
            # Subsequent layers in this block must use stride=1
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
    
class ResNet_34(nn.Module):
        """"""
        def __init__(self, in_channels: int = 3, num_classes: int = 1000, leaky: bool = False):
           super().__init__()
           self.leaky = leaky
           self.activation = LeakyReLU() if leaky else ReLU()
           
           # Padding 3 to properly reach output size 112 from a 224 input
           self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
           self.norm1 = BatchNorm(64)
           
           # Initial Max Pooling (3x3)
           self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

           # ResNet Layer-1 - Resolution [112, 112]
           # 6 identical 3x3 conv layers split into 3 blocks of 2 layers each (violet) -> 3 total blocks of 2 layers each
           self.layer1_1 = ResidualBlock(in_features=64, out_features=64, stride=1, num_layers=2, leaky=leaky)
           self.layer1_2 = ResidualBlock(in_features=64, out_features=64, stride=1, num_layers=2, leaky=leaky)
           self.layer1_3 = ResidualBlock(in_features=64, out_features=64, stride=1, num_layers=2, leaky=leaky)
           
           # ResNet Layer-2 - Resolution [56, 56]
           # 1 downsampling layer followed by 7 identical layers -> 4 total blocks of 2 layers each
           self.layer2_downsample = ResidualBlock(in_features=64, out_features=128, stride=2, num_layers=2, leaky=leaky)
           self.layer2_1 = ResidualBlock(in_features=128, out_features=128, stride=1, num_layers=2, leaky=leaky)
           self.layer2_2 = ResidualBlock(in_features=128, out_features=128, stride=1, num_layers=2, leaky=leaky)
           self.layer2_3 = ResidualBlock(in_features=128, out_features=128, stride=1, num_layers=2, leaky=leaky)

           # ResNet Layer-3 - Resolution [28, 28]
           # 1 downsampling layer followed by 11 identical layers -> 6 total blocks of 2 layers each
           self.layer3_downsample = ResidualBlock(in_features=128, out_features=256, stride=2, num_layers=2, leaky=leaky)
           self.layer3_1 = ResidualBlock(in_features=256, out_features=256, stride=1, num_layers=2, leaky=leaky)
           self.layer3_2 = ResidualBlock(in_features=256, out_features=256, stride=1, num_layers=2, leaky=leaky)
           self.layer3_3 = ResidualBlock(in_features=256, out_features=256, stride=1, num_layers=2, leaky=leaky)
           self.layer3_4 = ResidualBlock(in_features=256, out_features=256, stride=1, num_layers=2, leaky=leaky)
           self.layer3_5 = ResidualBlock(in_features=256, out_features=256, stride=1, num_layers=2, leaky=leaky)
           
           # ResNet Layer-4 - Resolution [14, 14]
           # 1 downsampling layer followed by 5 identical layers
           self.layer4_downsample = ResidualBlock(in_features=256, out_features=512, stride=2, num_layers=2, leaky=leaky)
           self.layer4_1 = ResidualBlock(in_features=512, out_features=512, stride=1, num_layers=2, leaky=leaky)
           self.layer4_2 = ResidualBlock(in_features=512, out_features=512, stride=1, num_layers=2, leaky=leaky)
           
           # Final Resolution - [7, 7]
           
           # GAP (Global Average Pooling - Calculates the average of all pixels in each channel) - 7x7 spatial size -> 1x1 vector
           self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
           
           self.fc34 = nn.Linear(512*1*1, num_classes)
           
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.activation(self.norm1(self.c1(x)))
            
            x = self.maxpool1(x)
            
            # Layer 1
            x = self.layer1_1(x)
            x = self.layer1_2(x)
            x = self.layer1_3(x)
            
            # Layer 2
            x = self.layer2_downsample(x)
            x = self.layer2_1(x)
            x = self.layer2_2(x)
            x = self.layer2_3(x)
            
            # Layer 3
            x = self.layer3_downsample(x)
            x = self.layer3_1(x)
            x = self.layer3_2(x)
            x = self.layer3_3(x)
            x = self.layer3_4(x)
            x = self.layer3_5(x)
            
            # Layer 4
            x = self.layer4_downsample(x)
            x = self.layer4_1(x)
            x = self.layer4_2(x)
            
            # Pool, Flatten, Classify
            x = self.avgpool2(x)
            x = torch.flatten(x, start_dim=1)
            x = self.fc34(x)
            
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
            
            print(f"ResNet_34 training will start on: {device.type.upper()}")
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
        
        DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ResNet34_model.pth')
        
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
            
            print(f"ResNet_34 model loaded from {path} to {device}")
        
        """Save model class method"""
        def save(self, path=None):
            """Saves the model's weights to a file."""
            path = path or self.DEFAULT_WEIGHTS
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
                
            torch.save(self.state_dict(), path)
            print(f"ResNet_34 model saved to {path}")
