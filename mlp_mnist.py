from mlp_complete import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


X_train = np.load('./data/mnist-train-x.npy')
y_train = np.load('./data/mnist-train-y.npy')
X_test = np.load('./data/mnist-test-x.npy')
y_test = np.load('./data/mnist-test-y.npy')

# we want to split into validation (20%)
val_ratio = 0.2
num_samples = X_train.shape[0]
num_val = int(num_samples * val_ratio)

# Shuffle indices
indices = np.random.permutation(num_samples)

# Split indices
train_indices = indices[num_val:]
val_indices = indices[:num_val]

# Create train/val splits
X_train_shuffled = X_train[train_indices]
y_train_shuffled = y_train[train_indices]
# validation sets
X_val = X_train[val_indices]
y_val = y_train[val_indices]
# reassign names
X_train = X_train_shuffled
y_train = y_train_shuffled

# Compute statistics for X (features)
X_mean = X_train.mean(axis=0)  # Mean of each feature
X_std = X_train.std(axis=0)  # Standard deviation of each feature

# Standardize X (only divide by standard deviation if standard deviation is NOT 0, otherwise will get NaNs)
X_train = np.where(X_std != 0, (X_train - X_mean) / X_std, 0)
X_val = np.where(X_std != 0, (X_val - X_mean) / X_std, 0)
X_test = np.where(X_std != 0, (X_test - X_mean) / X_std, 0)

# One hot encoding
y_train = np.eye(10)[y_train]
y_val = np.eye(10)[y_val]
y_test = np.eye(10)[y_test]

# Create a simple architecture (one hidden layer)
layers = (Layer(784, 64, RectifiedLinear()),
          Layer(64, 32, RectifiedLinear()),
          Layer(32, 10, Softmax()))

mlp = MultilayerPerceptron(layers=layers)

training_loss, validation_loss = mlp.train(X_train,
                                           y_train,
                                           CrossEntropy(),
                                           learning_rate=1E-3,
                                           batch_size=16,
                                           epochs=18,
                                           do_cat_acc=True,
                                           val_x=X_val,
                                           val_y=y_val,
                                           dropout=0.15,
                                           momentum=0.96,
                                           alpha=0.9)

test_loss, test_acc = mlp.test(X_test, y_test, CrossEntropy(), do_cat_acc=True)
print(f"Testing Loss: {test_loss:.6f}\nTesting Accuracy: {test_acc:.6f}")

# Plot the training and validation curves
plt.plot(training_loss, color='b', label="Training")
plt.plot(validation_loss, color='r', label="Validation")
plt.legend()
plt.title("MNIST MLP Loss Curves", size=16)
plt.xlabel("Epochs", size=16)
plt.ylabel("Cross Entropy", size=14)
plt.show()
