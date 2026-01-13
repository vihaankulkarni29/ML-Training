"""
1D Convolutional Neural Network for Antimicrobial Resistance Prediction
ResNet-1D Architecture with Residual Blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual Block for 1D CNN.
    
    Architecture:
        Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm -> Skip Connection -> ReLU
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of convolutional kernel
        stride (int): Stride for convolution
        padding (int): Padding for convolution
        downsample (nn.Module, optional): Downsample layer for skip connection
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.downsample = downsample
        
    def forward(self, x):
        """
        Forward pass with skip connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length)
        """
        identity = x
        
        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection and apply activation
        out += identity
        out = self.relu(out)
        
        return out


class DeepG2P(nn.Module):
    """
    Deep Genotype-to-Phenotype (DeepG2P) Model.
    1D ResNet for Antimicrobial Resistance Prediction.
    
    Architecture:
        - Initial Conv1d layer
        - Multiple Residual Blocks (ResNet backbone)
        - Global Average Pooling
        - Fully Connected Layer
        - Sigmoid Activation (Multi-label classification)
    
    Args:
        input_length (int): Length of input signal (default: 6000)
        input_channels (int): Number of input channels (default: 1)
        num_antibiotics (int): Number of antibiotics to predict (output size)
        base_channels (int): Base number of channels (default: 64)
        num_blocks (list): Number of residual blocks in each stage (default: [2, 2, 2, 2])
        dropout_rate (float): Dropout rate before final layer (default: 0.5)
    """
    
    def __init__(
        self,
        input_length=6000,
        input_channels=1,
        num_antibiotics=10,
        base_channels=64,
        num_blocks=[2, 2, 2, 2],
        dropout_rate=0.5
    ):
        super(DeepG2P, self).__init__()
        
        self.input_length = input_length
        self.input_channels = input_channels
        self.num_antibiotics = num_antibiotics
        
        # Initial convolution layer
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=base_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks (ResNet backbone)
        self.in_channels = base_channels
        self.layer1 = self._make_layer(base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, num_blocks[3], stride=2)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(base_channels * 8, num_antibiotics)
        
        # Sigmoid activation for multi-label classification
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, out_channels, num_blocks, stride):
        """
        Create a layer with multiple residual blocks.
        
        Args:
            out_channels (int): Number of output channels
            num_blocks (int): Number of residual blocks in this layer
            stride (int): Stride for the first block
            
        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        downsample = None
        
        # Create downsample layer if dimensions change
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        
        # First block with potential downsampling
        layers.append(
            ResidualBlock(
                self.in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                downsample=downsample
            )
        )
        
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, input_length)
            
        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, num_antibiotics)
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks (backbone)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        
        # Dropout
        x = self.dropout(x)
        
        # Fully connected layer
        x = self.fc(x)
        
        # Sigmoid activation
        x = self.sigmoid(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Extract feature maps from intermediate layers for visualization.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            dict: Dictionary of feature maps from different layers
        """
        features = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features['conv1'] = x.detach()
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features['layer1'] = x.detach()
        
        x = self.layer2(x)
        features['layer2'] = x.detach()
        
        x = self.layer3(x)
        features['layer3'] = x.detach()
        
        x = self.layer4(x)
        features['layer4'] = x.detach()
        
        return features


def create_deepg2p_model(input_length=6000, input_channels=1, num_antibiotics=10, model_size='medium'):
    """
    Factory function to create DeepG2P models of different sizes.
    
    Args:
        input_length (int): Length of input signal
        input_channels (int): Number of input channels
        num_antibiotics (int): Number of antibiotics to predict
        model_size (str): Model size - 'small', 'medium', or 'large'
        
    Returns:
        DeepG2P: Configured model instance
    """
    size_configs = {
        'small': {
            'base_channels': 32,
            'num_blocks': [2, 2, 2, 2],
            'dropout_rate': 0.3
        },
        'medium': {
            'base_channels': 64,
            'num_blocks': [2, 2, 2, 2],
            'dropout_rate': 0.5
        },
        'large': {
            'base_channels': 64,
            'num_blocks': [3, 4, 6, 3],
            'dropout_rate': 0.5
        }
    }
    
    config = size_configs.get(model_size, size_configs['medium'])
    
    return DeepG2P(
        input_length=input_length,
        input_channels=input_channels,
        num_antibiotics=num_antibiotics,
        **config
    )


if __name__ == "__main__":
    # Example usage and model testing
    print("=" * 60)
    print("DeepG2P Model Architecture")
    print("=" * 60)
    
    # Create model
    model = create_deepg2p_model(
        input_length=6000,
        input_channels=1,
        num_antibiotics=10,
        model_size='medium'
    )
    
    # Print model architecture
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 6000)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("Model created successfully!")
    print("=" * 60)
