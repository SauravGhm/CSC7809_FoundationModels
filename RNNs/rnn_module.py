import torch
import torch.nn as nn


class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModule, self).__init__()
        self.hidden_size = hidden_size

        # setup the weights
        self.Wxh = nn.Linear(input_size, hidden_size)  # Connect the input to the hidden state
        self.Whh = nn.Linear(hidden_size, hidden_size, bias=False)  # Connect the hidden state to itself
        self.Who = nn.Linear(hidden_size, output_size)  # Connect the hidden state to the output

        # Activation function
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        """
        Predict the hidden state
        :param x: Input at current timestep (batch_size, input_size)
        :param hidden: Previous hidden state (batch_size, hidden_size)
        :return: Output and new hidden state
        """
        hidden = self.tanh(self.Wxh(x) + self.Whh(hidden))  # Recurrent computation
        output = self.Who(hidden)  # Compute output
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state with zeros
        :param batch_size: Number of samples in the batch
        :return: Initial hidden state (batch_size, hidden_size)
        """
        return torch.zeros(batch_size, self.hidden_size)


# Some arbitrary parameters for the example
input_size = 10  # Number of input features
hidden_size = 20  # Number of hidden units
output_size = 5  # Output dimension
seq_len = 15  # Length of the input sequence
batch_size = 3  # Number of sequences in a batch

# Instantiate the RNN
rnn = RNNModule(input_size, hidden_size, output_size)

# Initialize hidden state to just zeros
hidden = rnn.init_hidden(batch_size)

# Arbitrary random input sequence
x_seq = torch.randn(batch_size, seq_len, input_size)

# Process sequence
outputs = []
for t in range(seq_len):
    x_t = x_seq[:, t, :]  # sample at current time step
    output, hidden = rnn(x_t, hidden)
    outputs.append(output)

# Convert output list to tensor (batch_size, seq_len, output_size)
outputs = torch.stack(outputs, dim=1)

print("Final output shape:", outputs.shape)
