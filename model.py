import torch
import torch.nn as nn
from torchvision import models
import torch

# Channel Attention Mechanism   
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel Attention
        avg_out = self.channel_attention[0:2](x)  # Avg pooling
        max_out = self.channel_attention[2:](x)  # Max pooling
        channel_attention = avg_out + max_out
        x = x * channel_attention
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Avg pooling along channel axis
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling along channel axis
        spatial_attention = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial_attention(spatial_attention)
        x = x * spatial_attention
        
        return x

class BirdClassifier(nn.Module):
    def __init__(self, num_classes=200):
        super(BirdClassifier, self).__init__()
        # Load the ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)

        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-2]  # Exclude avgpool and fc layers
        )
        for param in self.resnet.parameters():
            param.requires_grad = False
        in_features = 2048

        layers_to_unfreeze = list(self.resnet.children())[-4:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        # CBAM Module
        self.cbam = CBAM(in_features)

        # Custom Layers
        self.bn1 = nn.BatchNorm1d(in_features*7*7)  # Batch Normalization
        self.dropout1 = nn.Dropout(0.5)  # Dropout for regularization
        self.fc1 = nn.Linear(in_features*7*7, 1024)  # Custom Fully Connected Layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)  # Final Output Layer

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resnet(x)  # ResNet feature extraction
        x = self.cbam(x)  # CBAM Attention
        x = torch.flatten(x, start_dim=1)  # Flatten the output from CBAM

        # Apply custom layers
        x = self.bn1(x)  # Batch Normalization
        x = self.relu(x)
        x = self.fc1(x)  # First Fully Connected Layer
        x = self.relu(x)
        x = self.dropout1(x)  # Dropout for regularization
        x = self.fc2(x)  # Second Fully Connected Layer
        x = self.relu(x)
        x = self.fc3(x)  # Final Classification Layer
        return x
