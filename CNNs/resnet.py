import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    A simplified version of the BasicBlock used in ResNet-18/34, as introduced in:
    "Deep Residual Learning for Image Recognition" (https://arxiv.org/abs/1512.03385).

    Each block consists of:
      - 2D convolution (3x3) + BatchNorm + ReLU
      - 2D convolution (3x3) + BatchNorm
      - Optional downsampling (1x1 convolution with stride > 1 to match dimensions)

    The 'residual connection' (skip) adds the input to the block's output
    before applying ReLU again.

    Args:
        in_channels (int):  Number of input channels.
        out_channels (int): Number of output channels for this block.
        stride (int):       Stride for the first conv. If > 1, the spatial dimension is reduced.
    """
    expansion = 1  # Used in deeper Bottleneck blocks for ResNet-50/101/152; remains 1 for BasicBlock.

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # The first 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # The second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the input and output do not match in shape (channel or stride),
        # we need a 'downsample' layer to match them for the residual connection.
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass for the BasicBlock.

        Args:
            x (torch.Tensor): Shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: Shape (N, out_channels, H', W') after the block's transformations,
                          where H' and W' may be the same as H, W or smaller (if stride > 1).
        """
        identity = x  # This will be added to the output (residual connection).

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If there's a mismatch in shape, apply the downsample to identity
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection: add the (possibly downsampled) identity
        out += identity
        out = F.relu(out)

        return out


class SimpleResNet(nn.Module):
    """
    A minimal example of stacking BasicBlocks to form a small ResNet-like network.
    This example is not a standard ResNet-18/34, but shows how blocks can be chained.

    Model Flow:
        1) Initial convolution + BatchNorm + ReLU.
        2) A sequence of ResNet "layers", each containing multiple BasicBlocks.
        3) A global average pool and a final fully connected layer for classification.

    Args:
        num_classes (int): Number of output classes for the final classification layer.
    """

    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()

        # Initial stem: Typically in ResNet you might see a 7x7 conv with stride=2
        # but here we'll use a simpler 3x3.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Instead of a large stride here, we'll keep it simple
        # and do a straightforward approach.

        # Define layers (in a real ResNet, you'd have e.g. layer1, layer2, etc. with more blocks).
        self.layer1 = self._make_layer(in_channels=64, out_channels=64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(in_channels=64, out_channels=128, num_blocks=2, stride=2)
        # You could add more layers (layer3, layer4) for a deeper network.

        # Global average pool: reduce HxW to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Final linear layer: input channel size is 128 because after layer2, we have 128 channels.
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a stack of BasicBlocks, where the first block may have a different stride
        to handle downsampling (if needed).

        Args:
            in_channels (int):  Number of input channels to the first block in this layer.
            out_channels (int): Number of output channels for each block in this layer.
            num_blocks (int):   How many BasicBlocks to stack.
            stride (int):       The stride of the first block in this layer (for downsampling).

        Returns:
            nn.Sequential: A sequence of BasicBlock modules.
        """
        layers = []

        # The first block in this layer might downsample if stride != 1
        layers.append(BasicBlock(in_channels, out_channels, stride=stride))

        # For subsequent blocks in the same layer, stride=1
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initil stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Go through stacked layers
        x = self.layer1(x)
        x = self.layer2(x)

        # Global average pool -> flatten -> fully connected layer
        x = self.avgpool(x)  # shape: (N, 128, 1, 1)
        x = torch.flatten(x, 1)  # shape: (N, 128)
        x = self.fc(x)  # shape: (N, num_classes)
        return x


if __name__ == "__main__":
    # Create a random input of shape (batch_size=1, channels=3, height=224, width=224)
    x = torch.randn(1, 3, 224, 224)

    # Initialize the simple ResNet and run a forward pass
    model = SimpleResNet(num_classes=10)
    out = model(x)

    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {out.shape}")  # Should be (1, num_classes) = (1, 10) in this example
