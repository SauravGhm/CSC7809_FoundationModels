import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    A PyTorch implementation of AlexNet (without LRN).

    Architecture (for input 224x224):

    1) Convolution(3->64, kernel=11, stride=4, padding=2), ReLU
    2) MaxPool(kernel=3, stride=2)
    3) Convolution(64->192, kernel=5, padding=2), ReLU
    4) MaxPool(kernel=3, stride=2)
    5) Convolution(192->384, kernel=3, padding=1), ReLU
    6) Convolution(384->256, kernel=3, padding=1), ReLU
    7) Convolution(256->256, kernel=3, padding=1), ReLU
    8) MaxPool(kernel=3, stride=2)

    Followed by a classifier:

    1) Dropout
    2) Linear(256*6*6 -> 4096), ReLU
    3) Dropout
    4) Linear(4096 -> 4096), ReLU
    5) Linear(4096 -> num_classes)

    Args:
        num_classes (int): Number of classification categories.
                           Default = 1000 (as in ImageNet).
    """

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        # Feature extraction layers (the "convolutional" part)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Classifier layers (the "fully connected" part)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the AlexNet.

        Args:
            x (torch.Tensor): Shape (N, 3, H, W), typically (N, 3, 224, 224) for ImageNet.

        Returns:
            torch.Tensor: Logits of shape (N, num_classes).
        """
        x = self.features(x)  # Extract feature maps
        x = torch.flatten(x, 1)  # Flatten to (N, 256*6*6)
        x = self.classifier(x)  # Classify
        return x


if __name__ == "__main__":
    # Example usage: classify a random tensor
    # with batch size = 1 and image size = 224 x 224
    model = AlexNet(num_classes=10)  # For 10 classes instead of 1000
    x = torch.randn(1, 3, 224, 224)  # Simulate a single 224x224 RGB image
    output = model(x)
    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)  # Should be [1, 10]
