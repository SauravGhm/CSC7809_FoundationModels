import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple


def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    n_samples = train_x.shape[0]

    # Generate a random permutation of indices
    indices = np.random.permutation(n_samples)

    # Shuffle the arrays using the permutation
    shuffled_x = train_x[indices]
    shuffled_y = train_y[indices]

    # Create batches
    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        batch_x = shuffled_x[start_idx:end_idx]
        batch_y = shuffled_y[start_idx:end_idx]
        yield batch_x, batch_y


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: float) -> float:
        pass

    @abstractmethod
    def derivative(self, x: float) -> float:
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x: float) -> float:
        """

        :param x:
        :return:
        """
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: float) -> float:
        sig = self.forward(x)
        return sig * (1 - sig)


class Tanh(ActivationFunction):
    def forward(self, x: float) -> float:
        """
        Computes the hyperbolic tangent activation function.

        :param x: Input value
        :return: tanh(x)
        """
        return np.tanh(x)

    def derivative(self, x: float) -> float:
        """
        Computes the derivative of tanh(x), which is 1 - tanh^2(x).

        :param x: Input value
        :return: Derivative of tanh(x)
        """
        tanh_x = self.forward(x)
        return 1 - np.square(tanh_x)


class RectifiedLinear(ActivationFunction):
    def forward(self, x: np.ndarray) -> float:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> float:
        return (x > 0).astype(float)


class Linear(ActivationFunction):
    def forward(self, x: float) -> float:
        """
        Computes the linear activation function (identity function).

        :param x: Input value
        :return: x (identity function)
        """
        return x

    def derivative(self, x: float) -> float:
        """
        Computes the derivative of the linear activation function, which is always 1.

        :param x: Input value
        :return: 1 (since d/dx of x is 1)
        """
        return np.ones_like(x)


class Softmax(ActivationFunction):
    def forward(self, x: float) -> np.ndarray:
        """
        Computes the softmax activation function.

        :param x: Input value
        :return: softmax
        """
        # while it isn't necessary, you can get more stability by using this little trick
        exp_stbl = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_stbl / np.sum(exp_stbl, axis=-1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the softmax activation function

        :param x: Input value
        :return: derivative of softmax
        """
        batch_size, num_classes = x.shape

        # Compute batched Jacobian: S_ij = S_i * (δ_ij - S_j)
        jacobian = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            s_i = x[i].reshape(-1, 1)  # Column vector (num_classes, 1)
            jacobian[i] = np.diagflat(s_i) - (s_i @ s_i.T)

        return jacobian


class Softplus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the softplus cativaiton
        :param x: input value
        :return: softplus
        """
        return np.log1p(np.exp(x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of softplus
        :param x: input
        :return: softplus derivative (which is just the sigmoid)
        """
        return 1 / (1 + np.exp(-x))


class Mish(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Mish activation

        :param x: input
        :return: mish activation output
        """
        return x * np.tanh(np.log1p(np.exp(x)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the Mish activation
        :param x: input value
        :return: derivative of mish at x
        """
        softplus_x = np.log1p(np.exp(x))
        tanh_sp_x = np.tanh(softplus_x)
        sigmoid_x = 1 / (1 + np.exp(-x))

        return tanh_sp_x + x * sigmoid_x * (1 - tanh_sp_x**2)


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the squared error loss for each sample pair
        The 1/2 is just there to simplify the derivative

        :param y_true: true values of y
        :param y_pred: predicted values of y
        :return: squared error for each sample independently
        """
        return 1 / 2 * np.square(y_pred - y_true)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of squared error for each sample

        :param y_true: true values of y
        :param y_pred: predicted values of y
        :return: derivative of squared error for each sample independently
        """
        return (y_pred - y_true)


class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the cross entropy loss for each sample
        :param y_true: true values of y
        :param y_pred: predicted values of y
        :return: cross entropy loss for each sample independently
        """
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -np.sum(y_true * np.log(y_pred), axis=1)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the cross entropy loss for each sample
        :param y_true: true values of y
        :param y_pred: predicted values of y
        :return: derivative of cross entropy loss for each sample independently
        """
        # this clipping can help prevent NaNs in some edge cases
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / y_pred


class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Define a layer of neurons

        :param fan_in: number of neurons in previous layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of ActivationFunction to use for the layer
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        self.pre_activations = None
        self.activations = None
        self.delta = None

        # Use Glorot uniform for weight initialization
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        self.W = np.random.uniform(low=-limit, high=limit, size=(fan_in, fan_out))
        self.b = np.zeros(shape=(1, fan_out), dtype=float)

    def forward(self, h: np.ndarray, dropout: float = None):
        """
        Propagate output of previous layer through this layer
        :param h: output of previous layer
        :param dropout: dropout rate (0 to 1)
        :return: output of this layer
        """
        # compute the linear transform (affine transformation)
        self.pre_activations = h @ self.W + self.b
        # apply nonlinearity
        self.activations = self.activation_function.forward(self.pre_activations)

        if dropout is not None:
            # calculate the numberof neurons we are going to dropout
            num_dropouts = int(self.fan_out * dropout)
            # randomly sample (without replacement), the neurons to be dropped
            dropout_indcs = np.random.choice(np.arange(0, self.fan_out), size=num_dropouts, replace=False)
            # apply the same dropout across all batches
            self.activations[:, dropout_indcs] = 0.0

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray):
        """
        Backpropagate the error through this layer
        :param h: output of the previous layer
        :param delta: delta term (dL_do), computed from the next layer (previous backprop step)
        :return:
        """
        dphi_dz = self.activation_function.derivative(self.activations)
        if isinstance(self.activation_function, Softmax):
            # apply the dot product along the batch dimensions
            # (only used with Softmax, where dphi_dz is a matrix of Jacobians)
            dL_dz = np.einsum('bij, bj -> bi', dphi_dz, delta)
        else:
            dL_dz = delta * dphi_dz
        dL_dW = h.T @ dL_dz
        dL_db = np.sum(dL_dz, axis=0, keepdims=True)
        self.delta = dL_dz @ self.W.T
        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron from a list of layers
        :param layers: list of Layer instances
        """
        self.layers = layers

    def forward(self, x: np.ndarray, dropout: float=None) -> np.ndarray:
        """
        Propagate network input through the MLP
        :param x: network input of shape (batch size, num_features)
        :param dropout: dropout rate for the network
        :return: output of the network
        """
        i = 0
        for layer in self.layers:
            if i == 0:
                # for the first layer, the input is x
                hidden_rep = layer.forward(x)
            else:
                # only do dropout if dropout is set to a value AND this isn't the output layer
                dropout = dropout if dropout is not None and layer != self.layers[-1] else None
                # for all layers after the first one, the input is the output of the previous layer
                hidden_rep = layer.forward(hidden_rep, dropout=dropout)
            i += 1

        return hidden_rep

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray):
        """
        Backpropagat the loss gradient through the whole network, calculated dL/dW and dL/db for all layers
        :param loss_grad: derivative of the loss function wrt each sample in the batch
        :param input_data: input to the network
        :return: Tuple(weight grad list, bias grad list), Tuple of lists containing weight and bias gradients for each
            layer, in order from first to last layer
        """
        # these will store the weight and bias gradients for each layer (in backwards order)
        dl_dw_all = []
        dl_db_all = []

        delta = loss_grad
        for i in reversed(range(len(self.layers))):
            if i == 0:
                input_term = input_data
            else:
                input_term = self.layers[i - 1].activations
            dl_dw, dl_db = self.layers[i].backward(input_term, delta)
            delta = self.layers[i].delta

            dl_dw_all.append(dl_dw)
            dl_db_all.append(dl_db)

        return dl_dw_all[::-1], dl_db_all[::-1]

    def train(self,
              train_x: np.ndarray,
              train_y: np.ndarray,
              loss_func: LossFunction,
              learning_rate: float=1E-3,
              batch_size: int=16,
              epochs: int=32,
              do_cat_acc: bool = False,
              val_x: np.ndarray = None,
              val_y: np.ndarray = None,
              dropout: float=None,
              momentum: float=0.0,
              alpha: float=None):
        """
        Train the MLP
        :param train_x: training data inputs
        :param train_y: training data target outputs
        :param loss_func: instance of loss function
        :param learning_rate: learning rate to use
        :param batch_size: size of each batch
        :param epochs: number of training epochs
        :param do_cat_acc: True to calculat ecategorical accuracy for a classification dataset
        :param val_x: validation data inputs
        :param val_y: validation data target outputs
        :param dropout: dropout rate
        :param momentum: momentum rate
        :param alpha: RMSProp alpha rate
        :return: Tuple of loss for each epoch, (training loss, validation loss)
        """
        training_loss = []
        validation_loss = []
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            num_correct = 0
            # Momentum velocity terms
            velocity_w = None
            velocity_b = None
            # RMSProp accumulator
            accumulator_w = None
            accumulator_b = None

            for batch_num, (batch_x, batch_y) in enumerate(batch_generator(train_x, train_y, batch_size), 1):
                # get predictions on the batch
                predicted_y = self.forward(batch_x)
                # get the loss gradient for the batch, wrt the outputs
                loss_grad = loss_func.derivative(batch_y, predicted_y)
                # get the dL/dW and dL/db grads
                w_grads, b_grads = self.backward(loss_grad, batch_x)
                # initialize velocity to zero if it has not been set
                if velocity_w is None:
                    velocity_w = [np.zeros_like(w) for w in w_grads]
                    velocity_b = [np.zeros_like(b) for b in b_grads]
                # initialize RMSProp accumulator if it has not been set
                if accumulator_w is None:
                    accumulator_w = [np.full_like(w, fill_value=1E-8) for w in w_grads]
                    accumulator_b = [np.full_like(b, fill_value=1E-8) for b in b_grads]

                # update velocity and use that to update weights and biases for each layer
                for i in range(len(self.layers)):
                    # calculate exponential moving average of squared gradients for RMSProp
                    accumulator_w[i] = alpha * accumulator_w[i] + (1 - alpha) * w_grads[i] ** 2
                    accumulator_b[i] = alpha * accumulator_b[i] + (1 - alpha) * b_grads[i] ** 2

                    # this is for momentum. If momentum term is 0, then it is regular minibatch SGD
                    velocity_w[i] = momentum * velocity_w[i] + (1 - momentum) * w_grads[i]
                    velocity_b[i] = momentum * velocity_b[i] + (1 - momentum) * b_grads[i]

                    self.layers[i].W -= learning_rate / np.sqrt(accumulator_w[i]) * velocity_w[i] / batch_size
                    self.layers[i].b -= learning_rate / np.sqrt(accumulator_b[i]) * velocity_b[i] / batch_size

                # calculate loss after updating weights
                predicted_y = self.forward(batch_x, dropout=dropout)
                batch_loss = loss_func.loss(batch_y, predicted_y).mean()
                epoch_loss += batch_loss

                if do_cat_acc:
                    batch_predicted_class = predicted_y.argmax(axis=-1)
                    batch_true_class = batch_y.argmax(axis=-1)
                    num_correct += len(np.where(batch_predicted_class == batch_true_class)[0])

                num_batches += 1

            additional_text = ''

            if do_cat_acc:
                additional_text += f" Train Acc: {num_correct / len(train_x):.6f}"

            if val_x is not None:
                val_pred_y = self.forward(val_x)
                val_loss = loss_func.loss(val_y, val_pred_y).mean()
                validation_loss.append(val_loss)

                additional_text += f" Val Loss: {val_loss:.6f}"

                if do_cat_acc:
                    val_pred_class = val_pred_y.argmax(axis=-1)
                    val_true_class = val_y.argmax(axis=-1)
                    val_num_correct = len(np.where(val_pred_class == val_true_class)[0])

                    additional_text += f" Val Acc: {val_num_correct / len(val_x):.6f}"
            avg_epoch_loss = epoch_loss / num_batches
            training_loss.append(avg_epoch_loss)

            print(f"Epoch {epoch + 1} Train Loss: {avg_epoch_loss:.6f}" + additional_text)
        return training_loss, validation_loss

    def test(self,
             test_x: np.ndarray,
             test_y: np.ndarray,
             loss_func: LossFunction,
             do_cat_acc: bool=False):
        """
        Evaluate the model on a testing dataset
        :param test_x: testing inputs
        :param test_y: testing target outputs
        :param loss_func: Loss Function instance
        :param do_cat_acc: True to calculate categorical accuracy for classification tasks
        :return: Tuple of average loss and accuracy
        """
        predicted_y = self.forward(test_x)
        loss = loss_func.loss(test_y, predicted_y).mean()
        acc = None

        # calculate categorical accuracy
        if do_cat_acc:
            pred_class = predicted_y.argmax(axis=-1)
            true_class = test_y.argmax(axis=-1)
            num_correct = len(np.where(pred_class == true_class)[0])

            acc = num_correct / len(test_x)

        return loss, acc
