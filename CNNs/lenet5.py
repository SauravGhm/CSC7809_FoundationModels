import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# Define the LeNet-5 Model
class LeNet5(nn.Module):
    def __init__(self):
        """
        Defines a LeNet 5 architecture
        """
        super(LeNet5, self).__init__()
        # first conv layer uses 6 5x5 kernels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)  # Output: 28x28x6
        # define the activation func. we are using (Rectified Linear Unit)
        self.relu = nn.ReLU()
        # Define our downsampling module using 2x2 average pooling
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Output: 14x14x6
        # second convolutional layer with 16 5x5 kernels
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)  # Output: 10x10x16
        # third convolutional layer with 120 5x5 kernels
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)  # Output: 10x10x16
        # MLP
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)  # Output: 10 classes (digits 0-9)

    def forward(self, x):
        """
        Define t
        :param x:
        :return:
        """
        x = self.pool(self.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(self.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = self.relu(self.conv3(x))  # Conv3 -> ReLU
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))  # Fully Connected 1
        x = self.fc2(x)  # Output Layer (No activation needed, as CrossEntropyLoss applies Softmax)
        return x


# Load MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST mean & std
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")

# Testing the Model
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
