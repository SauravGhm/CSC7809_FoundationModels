import torch
import sentencepiece as spm
import os

from gru_model import GRULanguageModel
from utils import load_tokenizer


MODEL_PATH = "gru_model.pt"
TOKENIZER_PATH = "../bpe_tokenizer.model"
tokenizer = load_tokenizer(TOKENIZER_PATH)
vocab_size = tokenizer.get_piece_size()

# Recreate the model architecture
model = GRULanguageModel(
    vocab_size=vocab_size,
    embed_dim=512,
    hidden_dim=1024,
    num_layers=6
)

# Load the trained weights
model.load_state_dict(torch.load("gru_model.pt"))
model.eval()

prompt = "The quick brown fox "
output = model.generate(
    tokenizer=tokenizer,
    prompt=prompt,
    max_length=512,
    eos_token_id=tokenizer.piece_to_id("<eos>"),
    temperature=1.0,  # lower = more deterministic
    device="cuda" if torch.cuda.is_available() else "cpu"
)

print("Generated output:", output)

