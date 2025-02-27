from mlp_complete import *


from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# downlod the dataset
auto_mpg = fetch_ucirepo(id=9)

# data (these are pandas dataframes)
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Combine features and target into a single DataFrame for filterng
data = pd.concat([X, y], axis=1)

# Drop rows where the target variable is NaN. These will break the code otherwise
cleaned_data = data.dropna()

# Split the data back into features (X) and target (y)
X = cleaned_data.iloc[:, :-1].to_numpy()
y = cleaned_data.iloc[:, -1].to_numpy().reshape(-1, 1)

from sklearn.model_selection import train_test_split

# Do a 70/30 split
X_train, X_leftover, y_train, y_leftover = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,  # for reproducibility
    shuffle=True,  # whether to shuffle the data before splitting
)

# Split the remaining 30% into validation/testing (15%/15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_leftover, y_leftover,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)

# Compute statistics for X
X_mean = X_train.mean(axis=0)  # Mean of each feature
X_std = X_train.std(axis=0)  # Standard deviation of each feature

# Standardize X
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Compute statistics for y
y_mean = y_train.mean()  # Mean of target
y_std = y_train.std()  # Standard deviation of target

# Standardize y
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

# Construct a simple network
layers = (Layer(7, 3, Sigmoid()), Layer(3, 1, Linear()))
mlp = MultilayerPerceptron(layers=layers)
# Train the model for regression
training_loss, validation_loss = mlp.train(X_train,
                                           y_train,
                                           SquaredError(),
                                           learning_rate=1E-1,
                                           batch_size=16,
                                           epochs=50,
                                           val_x=X_val,
                                           val_y=y_val,
                                           momentum=0.9,
                                           alpha=0.9)

test_loss, _ = mlp.test(X_test, y_test, SquaredError())
print(f"Testing Loss: {test_loss:.6f}")

# Plot the training and validation curves
plt.plot(training_loss, color='b', label="Training")
plt.plot(validation_loss, color='r', label="Validation")
plt.legend()
plt.title("Vehicle MPG MLP Loss Curves", size=16)
plt.xlabel("Epochs", size=16)
plt.ylabel("Mean Squared Error", size=14)
plt.show()
