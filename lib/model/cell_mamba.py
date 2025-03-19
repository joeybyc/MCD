import torch
import torch.nn as nn
import torch.nn.functional as F

class SS2D(nn.Module):
    """
    2D Selective Scan module that processes spatial information in 4 directions.
    Projects input features, applies directional scanning, and merges results.
    
    Args:
        dim (int): Input feature dimension
    """
    def __init__(self, dim):
        super(SS2D, self).__init__()
        self.dim = dim
        # Project input to higher dimension for richer feature interaction
        self.proj_in = nn.Linear(dim, dim * 2)
        # Project back to original dimension after scanning
        self.proj_out = nn.Linear(dim * 2, dim)
    
    def forward(self, x):
        # x shape: [batch_size, height, width, channels]
        B, H, W, C = x.shape
        x = self.proj_in(x)  # Expand channel dimension: C -> C*2
        
        # Create tensors for 4 scanning directions
        x_lr = x  # Left to right scan
        x_rl = x.flip(2)  # Right to left scan (flip width dimension)
        x_tb = x.permute(0, 2, 1, 3)  # Top to bottom scan
        x_bt = x_tb.flip(2)  # Bottom to top scan
        
        # Apply selective scan operation using einsum
        # This maintains spatial information while allowing feature interaction
        x_lr = torch.einsum('bhwc,bhwc->bhwc', x_lr, x_lr)
        x_rl = torch.einsum('bhwc,bhwc->bhwc', x_rl, x_rl)
        x_tb = torch.einsum('bhwc,bhwc->bhwc', x_tb, x_tb)
        x_bt = torch.einsum('bhwc,bhwc->bhwc', x_bt, x_bt)
        
        # Merge all scanning directions
        # 1. Flip back the reversed scans
        # 2. Average all directions to combine information
        x = (x_lr + x_rl.flip(2) + x_tb.permute(0, 2, 1, 3) + x_bt.permute(0, 2, 1, 3)) / 4
        x = self.proj_out(x)  # Project back to original dimension: C*2 -> C
        return x

class VSSBlock(nn.Module):
    """
    Visual State-Space (VSS) Block based on VMamba.
    """
    def __init__(self, dim):
        super(VSSBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        x = x + self.ss2d(self.norm1(x))  # Apply SS2D
        x = x + self.ffn(self.norm2(x))   # Apply Feed-forward Network
        return x


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
    """
    Modified network using VSS Blocks instead of CNN layers, while keeping the spatial attention.
    Designed for 20x20 image classification.
    """
    def __init__(self, size_img=20, dim=32):
        super(Net, self).__init__()
        self.size_img = size_img
        self.dim = dim
        
        # Input embedding
        self.embedding = nn.Linear(1, dim)
        
        # VSS Blocks replacing CNNs
        self.vss1 = VSSBlock(dim)
        self.vss2 = VSSBlock(dim)
        # self.vss3 = VSSBlock(dim)
        
        # Spatial Attention remains unchanged
        self.attention = SpatialAttention()
        
        # Feature processing
        self.norm = nn.LayerNorm(dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(1)
        
        # Embedding
        x = x.permute(0, 2, 3, 1)  # Reshape to (B, H, W, C)
        x = self.embedding(x)
        
        # Apply VSS Blocks
        x = self.vss1(x)
        x = self.vss2(x)
        # x = self.vss3(x)
        
        # Apply spatial attention (original position)
        x = x.permute(0, 3, 1, 2)  # Convert back to (B, C, H, W)
        x = self.attention(x)
        
        # Final classification
        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
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
