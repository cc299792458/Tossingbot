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
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
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
        self.rb1 = ResidualBlock(640, 256)  # 512 dim visual feature + 128 dim estimated velocity
        self.rb2 = ResidualBlock(256, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rb3 = ResidualBlock(128, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # NOTE: 2 channels here represent "grasp" or "not grasp", corresponding to a supervised learning manner.
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
        self.rb1 = ResidualBlock(640, 256)  # 512 dim visual feature + 128 dim estimated velocity
        self.rb2 = ResidualBlock(256, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rb3 = ResidualBlock(128, 64)
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

if __name__ == '__main__':
    # Create instances of each module
    perception_module = PerceptionModule()
    grasping_module = GraspingModule()
    throwing_module = ThrowingModule()
    
    # Sample input tensor (Batch size 1, 4 channels, 180x140 image)
    input_tensor = torch.randn(1, 4, 180, 140)

    physics_velocity = 3.5  # Example estimated velocity

    # Pass input through Perception Module
    perception_output = perception_module(input_tensor)
    print(f'Perception Output Shape: {perception_output.shape}')  # Expected shape: [1, 512, H, W]

    # Create a velocity image (B, 128, H, W) with each pixel holding the value of the estimated velocity
    B, C, H, W = perception_output.shape
    velocity_image = torch.full((B, 128, H, W), physics_velocity, device=perception_output.device)

    # Concatenate the perception output and the velocity image
    perception_with_velocity = torch.cat([perception_output, velocity_image], dim=1)  # New shape: [1, 640, H, W]

    # Pass concatenated output to Grasping and Throwing Modules
    grasping_output = grasping_module(perception_with_velocity)
    throwing_output = throwing_module(perception_with_velocity)

    print(f'Grasping Output Shape: {grasping_output.shape}')  # Expected output: [1, 2, H', W']
    print(f'Throwing Output Shape: {throwing_output.shape}')  # Expected output: [1, 1, H', W']