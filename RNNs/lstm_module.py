import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        # Input, Forget, Cell, and Output gate weight matrices

        # Connects the input to the input gate output
        self.Wxi = nn.Linear(input_size, hidden_size)
        # Connects the previous hidden state to the input gate output
        self.Whi = nn.Linear(hidden_size, hidden_size, bias=False)

        # Connects the input to the forget gate output
        self.Wxf = nn.Linear(input_size, hidden_size)
        # Connects the previous hidden state to the forget gate output
        self.Whf = nn.Linear(hidden_size, hidden_size, bias=False)

        # Connects the input to the memory cell (input node, in ref. to class slides) output
        self.Wxc = nn.Linear(input_size, hidden_size)
        # Connects the previous hidden state to the memory cell output
        self.Whc = nn.Linear(hidden_size, hidden_size, bias=False)

        # Connects the input to the output gate's output
        self.Wxo = nn.Linear(input_size, hidden_size)
        # Connects the previous hidden state to the output gate's output
        self.Who = nn.Linear(hidden_size, hidden_size, bias=False)

        # Simple linear layer that computes the LSTM module's final output (only used in the final LSTM module)
        self.Why = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden_state, cell_state):
        """
        Compute the hidden state, memory cell internal state, and output of the LSTM module
        :param x: Input at current timestep (batch_size, input_size)
        :param hidden: Previous hidden state (batch_size, hidden_size)
        :param cell: Previous cell state (batch_size, hidden_size)
        :return: Output, new hidden state, new cell state
        """
        # Compute gate outputs
        input_gate = self.sigmoid(self.Wxi(x) + self.Whi(hidden_state))
        forget_gate = self.sigmoid(self.Wxf(x) + self.Whf(hidden_state))
        output_gate = self.sigmoid(self.Wxo(x) + self.Who(hidden_state))

        # Compute the cell node output (candidate cell state)
        cell_node = self.tanh(self.Wxc(x) + self.Whc(hidden_state))

        # Update cell state
        cell_state = forget_gate * cell_state + input_gate * cell_node

        # Update hidden state
        hidden_state = output_gate * self.tanh(cell_state)

        # Compute final output of the network
        output = self.Why(hidden_state)

        return output, hidden, cell

    def init_hidden_cell(self, batch_size):
        """
        Initializes hidden and cell states to just zeros
        :param batch_size: Number of samples in the batch
        :return: Initial hidden state, Initial cell state (both of shape: batch_size x hidden_size)
        """
        return torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size)


# Some arbitrary parameters for the example
input_size = 10   # Number of input features
hidden_size = 20  # Number of hidden units
output_size = 5   # Output dimension
seq_len = 15      # Length of the input sequence
batch_size = 3    # Number of sequences in a batch

# Instantiate the LSTM
lstm = LSTM(input_size, hidden_size, output_size)

# Initialize hidden and memory cell states to just zeros
hidden, cell = lstm.init_hidden_cell(batch_size)

# Arbitrary random input sequence
x_seq = torch.randn(batch_size, seq_len, input_size)

# Process sequence
outputs = []
for t in range(seq_len):
    x_t = x_seq[:, t, :]  # sample at current time step
    output, hidden, cell = lstm(x_t, hidden, cell)
    outputs.append(output)

# Convert output list to tensor (batch_size, seq_len, output_size)
outputs = torch.stack(outputs, dim=1)

print("Final output shape:", outputs.shape)
