import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If in_channels != out_channels, apply 1x1 convolution to residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply 1x1 convolution if necessary to match the number of channels
        if self.residual_conv is not None:
            residual = self.residual_conv(x)

        out += residual  # Residual connection
        out = F.relu(out)
        return out

class PerceptionModule(nn.Module):
    def __init__(self):
        super(PerceptionModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rb1 = ResidualBlock(64, 128)  # From 64 channels to 128
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rb2 = ResidualBlock(128, 256)  # From 128 channels to 256
        self.rb3 = ResidualBlock(256, 512)  # From 256 channels to 512

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.rb1(x)
        x = self.pool2(x)
        x = self.rb2(x)
        x = self.rb3(x)
        return x

class GraspingModule(nn.Module):
    def __init__(self):
        super(GraspingModule, self).__init__()
        self.rb1 = ResidualBlock(256)
        self.rb2 = ResidualBlock(128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rb3 = ResidualBlock(64)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.up1(x)
        x = self.rb3(x)
        x = self.up2(x)
        x = self.conv_final(x)
        return x

class ThrowingModule(nn.Module):
    def __init__(self):
        super(ThrowingModule, self).__init__()
        self.rb1 = ResidualBlock(256)
        self.rb2 = ResidualBlock(128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rb3 = ResidualBlock(64)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.up1(x)
        x = self.rb3(x)
        x = self.up2(x)
        x = self.conv_final(x)
        return x

# Example usage
if __name__ == '__main__':
    # Create instances of each module
    perception_module = PerceptionModule()
    grasping_module = GraspingModule()
    throwing_module = ThrowingModule()
    
    # Sample input tensor (Batch size 1, 3 channels, 224x224 image)
    input_tensor = torch.randn(1, 3, 224, 224)

    # Pass input through Perception Module
    perception_output = perception_module(input_tensor)

    # Pass perception output to Grasping and Throwing Modules
    grasping_output = grasping_module(perception_output)
    throwing_output = throwing_module(perception_output)

    print(f'Grasping Output Shape: {grasping_output.shape}')
    print(f'Throwing Output Shape: {throwing_output.shape}')
