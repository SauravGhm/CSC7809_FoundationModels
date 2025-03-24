import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sentencepiece as spm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

from gru_model import GRULanguageModel
from utils import load_tokenizer

# ---------------------------
# Handle the tokenizer
# ---------------------------
MODEL_PATH = "gru_model.pt"
TOKENIZER_PATH = "../bpe_tokenizer.model"
tokenizer = load_tokenizer(TOKENIZER_PATH)
eos_token_id = tokenizer.piece_to_id('<eos>')


TRAIN_FILE = "../data/train.jsonl"
VAL_FILE = "../data/test.jsonl"
VOCAB_SIZE = tokenizer.get_piece_size()
BATCH_SIZE = 64
EPOCHS = 100
EMBED_DIM = 512
HIDDEN_DIM = 1024
NUM_LAYERS = 6
LEARNING_RATE = 3e-4
PATIENCE = 3
MAX_SEQ_LEN = 128  # truncate long samples


class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_seq_len=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item["prompt"] + " " + item["completion"]
                token_ids = tokenizer.encode(text, out_type=int)[:max_seq_len]
                if len(token_ids) < 2:
                    continue
                self.samples.append(token_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids


def collate_fn(batch):
    input_batch, target_batch = zip(*batch)
    input_batch = nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=0)
    target_batch = nn.utils.rnn.pad_sequence(target_batch, batch_first=True, padding_value=0)
    return input_batch, target_batch

# ---------------------------
# Training & Validation
# ---------------------------
def train_model():
    device = torch.device("mps")

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    vocab_size = tokenizer.get_piece_size()

    train_dataset = TextDataset(TRAIN_FILE, tokenizer, MAX_SEQ_LEN)
    val_dataset = TextDataset(VAL_FILE, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = GRULanguageModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float('inf')
    no_improve_epochs = 0

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for input_ids, target_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                logits, _ = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("✅ Model improved and saved.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print("⏹ Early stopping triggered.")
                break

    # Plot Loss Curves
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()


if __name__ == "__main__":
    train_model()