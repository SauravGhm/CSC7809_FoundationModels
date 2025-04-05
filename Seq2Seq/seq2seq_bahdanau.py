import torch
import torch.nn as nn

from RNNs.gru_module import GRUModule


class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size

        # setup the weights
        self.Wxr = nn.Linear(input_size, hidden_size)  # Connect the input to the reset gate
        self.Whr = nn.Linear(hidden_size, hidden_size, bias=False)  # Connect the prior hidden state to the reset gate

        self.Wxz = nn.Linear(input_size, hidden_size)  # Connect the input to the update gate
        self.Whz = nn.Linear(hidden_size, hidden_size, bias=False)  # Connect the prior hidden state to the update gate

        self.Wxh = nn.Linear(input_size, hidden_size)  # Connect the input to the candidate hidden state
        self.Whh = nn.Linear(hidden_size, hidden_size,
                             bias=False)  # Connect the hidden state to the candidate hidden state

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        """
        Predict the hidden state
        :param x: Input at current timestep (batch_size, input_size)
        :param hidden: Previous hidden state (batch_size, hidden_size)
        :return: Output and new hidden state
        """
        # compute the reset gate
        reset_gate = self.sigmoid(self.Wxr(x) + self.Whr(hidden))

        # Compute the update gate
        update_gate = self.sigmoid(self.Wxz(x) + self.Whz(hidden))

        # compute the candidate hidden state (applying the reset gate)
        candidate_hidden = self.tanh(self.Wxh(x) + self.Whh(reset_gate * hidden))

        # apply the update gate to determine the new hidden state
        hidden = update_gate * hidden + (1 - update_gate) * candidate_hidden

        return hidden

    def init_hidden(self, batch_size):
        """
        Initializes the hidden state with zeros
        :param batch_size: Number of samples in the batch
        :return: Initial hidden state (batch_size, hidden_size)
        """
        return torch.zeros(batch_size, self.hidden_size)


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, energy_size):
        super(BahdanauAttention, self).__init__()

        # W1, W2, and v are the learnable parameters, setup as simple fully connected layers
        self.W1 = nn.Linear(encoder_hidden_size, energy_size)
        self.W2 = nn.Linear(decoder_hidden_size, energy_size, bias=False)
        self.v = nn.Linear(energy_size, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, encoder_hidden_states, decoder_hidden):
        """
        Compute the context vector and attention weights
        :param encoder_hidden_states: The sequences of hidden states computed by the encoder
        :param decoder_hidden: The decoder's prior hidden state
        :return: context vector, attention weights
        """
        # the dimensions of the inputs will look like these:
        #   encoder_hidden_states: (batch size, seq_len, hidden size)
        #   decoder_hidden: (batch size, hidden size)
        input_seq_len = encoder_hidden_states.size(1)

        # we start with the decoder_hidden state computed over the batch, with shape (batch size, hidden size)
        # in the energy calculation, we need to add the transformed decoder hidden state to each of the transformed
        # encoder hidden states, without having to loop, so we reshape and broadcast the decoder hidden state to the
        # same shape as the encoder hidden states tensor

        # unsqueeze(1) adds a new dimension at index 1, so we go from (batch size, hidden size) to (batch size, 1, hidden size)
        # expand(-1, input_seq_len, -1) copies the decoder_hidden across the seq_len dimension, the -1s keep those dims
        # unchanged
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, input_seq_len, -1)
        # compute energy
        energy = self.tanh(self.W1(encoder_hidden_states) + self.W2(decoder_hidden_expanded))
        # compute scores (collapse 1st dimension with squeeze(1))
        scores = self.v(energy).squeeze(-1)  # (batch size, seq_len)
        # apply softmax to attention scores for each batch independently
        attn_weights = torch.softmax(scores, dim=1)
        # compute the context vector (torch.bmm is batch matrix multiply). In this case, attn_weight is of shape
        # (batch size, seq_len) and encoder_hidden_states is (batch size, seq_len, hidden size)
        # we first do unsequeeze(1) to make attn_weight be (batch size, 1, seq_len), then torch.bmm
        # will do a matrix multiply for each batch, essentially each one is a (1, seq_len) X (seq_len, hidden size)
        # operation, resulting in a (batch size, 1, hidden size), which we collapse to (batch size, hidden size) using
        # squeeze(1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_hidden_states).squeeze(1)

        return context, attn_weights


class DecoderGRUWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, energy_size):
        super(DecoderGRUWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = BahdanauAttention(hidden_size, hidden_size, energy_size)

        # setup fully connected layers for computing reset gate
        self.Wxr = nn.Linear(input_size + hidden_size, hidden_size)
        self.Whr = nn.Linear(hidden_size, hidden_size, bias=False)

        # setup fully connected layers for computing update gate
        self.Wxz = nn.Linear(input_size + hidden_size, hidden_size)
        self.Whz = nn.Linear(hidden_size, hidden_size, bias=False)

        # setup fully connected layers for computing candidate hidden state
        self.Wxh = nn.Linear(input_size + hidden_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Who = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden, encoder_outputs):
        # Compute Bahdanau attention
        context, attn_weights = self.attention(encoder_outputs, hidden)
        # Our input now becomes to concatenated input and context vector
        x_context = torch.cat([x, context], dim=-1)

        # compute GRU gates
        reset_gate = self.sigmoid(self.Wxr(x_context) + self.Whr(hidden))
        update_gate = self.sigmoid(self.Wxz(x_context) + self.Whz(hidden))
        # compute candidate hidden state
        h_tilde = self.tanh(self.Wxh(x_context) + self.Whh(reset_gate * hidden))
        # compute the hidden state of the decoder
        hidden = update_gate * hidden + (1 - update_gate) * h_tilde

        # compute the decoder output
        output = self.Who(hidden)

        return output, hidden, attn_weights


# Setup some parameters just as an example
input_size = 10  # this is the size of our input (which would be the embedding vector size, for example)
hidden_size = 20  # size of our hidden state, in practice this woul dbe bigger
output_size = 5  # size of our output (size of our vocabulary for a lang model)
energy_size = 15  # this is the size of the energy layer in the MLP that computes the attention score
seq_len = 12  # length of our example sequence
batch_size = 4

# Encoder and decoder initialization
encoder = GRUEncoder(input_size, hidden_size)
decoder = DecoderGRUWithAttention(input_size, hidden_size, output_size, energy_size)

# Create some dummy data
x_seq = torch.randn(batch_size, seq_len, input_size)
# initialize the encoder's hidden state
hidden = encoder.init_hidden(batch_size)

# Encode the input sequence
encoder_hidden_states = []
for t in range(seq_len):
    hidden = encoder(x_seq[:, t, :], hidden)
    encoder_hidden_states.append(hidden.unsqueeze(1))

# convert the list of encoder hidden states, each having a shape of (batch size, 1, hidden size) to a single
# tensor of shape (batch size, seq_len, hidden size) by concatenating each tensor in the sequence dimension
encoder_hidden_states = torch.cat(encoder_hidden_states, dim=1)

# Decode (simulate teacher forcing, where we would know the target sequence that we want the model to produce)
# the decoder's initial hidden state is the hidden state of the last step in the encoder
decoder_hidden = encoder_hidden_states[:, -1, :]
# random input sequence
decoder_inputs = torch.randn(batch_size, seq_len, input_size)

decoder_outputs = []
for t in range(seq_len):
    # this is the target decoder output at this time step which we use for teacher forcing as the input
    x_t = decoder_inputs[:, t, :]
    # compute the model's predicted output, its updated hidden state, and the attention weights
    output, decoder_hidden, attn_weights = decoder(x_t, decoder_hidden, encoder_hidden_states)
    # keep track of predicted output
    decoder_outputs.append(output)

decoder_outputs = torch.stack(decoder_outputs, dim=1)
print("Decoder output shape:", decoder_outputs.shape)
