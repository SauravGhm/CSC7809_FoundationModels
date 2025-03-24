import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=6, dropout=0.2, pad_token_id=0):
        super(GRULanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            hidden: optional, GRU hidden state (num_layers, batch_size, hidden_dim)
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        embeds = self.embedding(input_ids)
        output, hidden = self.gru(embeds, hidden)
        logits = self.fc(output)
        return logits, hidden

    def predict_next_token(self, input_ids, temperature=1.0):
        """
        Predict the next token ID from the last token in input_ids.
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.argmax(probs, dim=-1)  # Greedy decoding (Undergrad)
            return next_token_id.item()

    def generate(self, tokenizer, prompt, max_length=50, eos_token_id=None, temperature=1.0, device='cpu'):
        """
        Generate text given a prompt.
        """
        self.eval()
        input_ids = tokenizer.encode(prompt, out_type=int)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated_ids = input_ids.copy()
        hidden = None

        for _ in range(max_length):
            logits, hidden = self.forward(input_tensor, hidden)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            next_token_id = torch.argmax(probs, dim=-1).item()

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            generated_ids.append(next_token_id)
            input_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

        return tokenizer.decode(generated_ids, out_type=str)
