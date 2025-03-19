import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Generate attention map
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.conv(attention)
        
        # Apply attention
        return x * attention_map




class Net(nn.Module):
    def __init__(self, size_img=20):
        super(Net, self).__init__()
        self.size_img = size_img
        
        # Initial convolution layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 32 x 10 x 10
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 64 x 5 x 5
        )
        
        # Spatial attention module
        self.attention = SpatialAttention()
        
        # Feature processing after attention
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()  # Output: 128 x 5 x 5
        )
        
        # Global average pooling and classification layers
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # Ensure input is properly shaped (B, 1, L, L)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(1)
            
        # Initial convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Apply spatial attention
        x = self.attention(x)
        
        # Further processing
        x = self.conv3(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x

# Forward test
if __name__ == "__main__":
    model = Net()
    sample_input = torch.randn(8, 1, 20, 20)  # Batch size 8, single-channel 20x20 image
    output = model(sample_input)
    print("Output shape:", output.shape)  # Expecting (8, 2) for binary classification
        # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Convert to millions for better readability
    total_params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6
    
    print(f"\nModel Parameters:")
    print(f"Total Parameters: {total_params:,} ({total_params_m:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params_m:.2f}M)")