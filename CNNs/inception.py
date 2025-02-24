import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """
    An Inception module based on the architecture introduced in GoogLeNet:
    https://arxiv.org/abs/1409.4842

    The module consists of four parallel branches:
        1. A 1x1 convolution
        2. A 1x1 convolution followed by a 3x3 convolution
        3. A 1x1 convolution followed by a 5x5 convolution
        4. A 3x3 max pool followed by a 1x1 convolution

    All branches' outputs are concatenated along the channel dimension.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_1x1 (int): Number of output channels for the 1x1 convolution branch.
        red_3x3 (int): Number of output channels for the 1x1 "reduction" in the 3x3 branch.
        out_3x3 (int): Number of output channels for the 3x3 convolution.
        red_5x5 (int): Number of output channels for the 1x1 "reduction" in the 5x5 branch.
        out_5x5 (int): Number of output channels for the 5x5 convolution.
        pool_proj (int): Number of output channels for the 1x1 convolution after the max pool.
    """

    def __init__(
            self,
            in_channels: int,
            out_1x1: int,
            red_3x3: int,
            out_3x3: int,
            red_5x5: int,
            out_5x5: int,
            pool_proj: int
    ):
        super(InceptionModule, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 convolution -> 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 convolution -> 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        # Branch 4: MaxPool -> 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass for the Inception module.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W)

        Returns:
            torch.Tensor: Concatenated result of the four branches
                         along the channel dimension,
                         shape (N, sum_of_all_out_channels, H, W)
        """
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        return torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], 1)


class SimpleInceptionNet(nn.Module):
    """
    A simple example network that demonstrates how to stack multiple
    Inception modules. This is not a full GoogLeNet, but it shows
    the principle of chaining Inception modules in a sequence.
    """

    def __init__(self, num_classes=10):
        super(SimpleInceptionNet, self).__init__()

        # Initial convolution to reduce the spatial size a bit
        # and match the number of input channels.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # First Inception module
        self.inception1 = InceptionModule(
            in_channels=64,
            out_1x1=32,
            red_3x3=32,
            out_3x3=64,
            red_5x5=8,
            out_5x5=16,
            pool_proj=16
        )

        # Second Inception module
        # Note: The output channels of inception1 = (32 + 64 + 16 + 16) = 128
        # This value becomes the in_channels for the next Inception.
        self.inception2 = InceptionModule(
            in_channels=128,
            out_1x1=64,
            red_3x3=64,
            out_3x3=96,
            red_5x5=16,
            out_5x5=32,
            pool_proj=32
        )

        # Another pooling to reduce spatial dimensions further
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        # A fully connected layer for classification
        # The output from inception2 has 64 + 96 + 32 + 32 = 224 channels
        self.fc = nn.Linear(224, num_classes)

    def forward(self, x):
        # Initial conv + pool
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Pass through the first Inception module
        x = self.inception1(x)

        # Pass through the second Inception module
        x = self.inception2(x)

        # Global average pooling to get a 1x1 spatial dimension
        x = self.pool2(x)

        # Flatten and feed into a linear layer for classification
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Test the model with a random input of size (batch=1, channels=3, height=224, width=224)
    # This is a typical image input size for many networks, but note we do
    # a 7x7 conv with stride=2 and then a 3x3 pooling with stride=2, so the
    # spatial dimensions will shrink quickly.
    model = SimpleInceptionNet(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print("Input shape:  ", x.shape)
    print("Output shape: ", output.shape)
    # Output shape should be [1, 10], indicating 10 class scores.
