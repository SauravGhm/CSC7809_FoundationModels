import os
import json
import random
import re

# Set seed for reproducibility
random.seed(42)

# Define paths
DATA_DIR = "./data/raw"
OUTPUT_DIR = "./data"
TRAIN_RATIO = 0.8  # 80% training, 20% testing
MIN_SEQ_LENGTH = 5  # Minimum words in a prompt
MAX_SEQ_LENGTH = 128  # Maximum words in a prompt

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Function to clean and preprocess text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9,.!?;:'\"\s]", "", text)  # Remove special characters
    text = text.replace("\n", " ")  # Replace newlines with spaces
    text = " ".join(text.split())  # Normalize spacing
    return text


# Function to extract content between "START OF THE PROJECT GUTENBERG" and "END OF THE PROJECT GUTENBERG"
def extract_gutenberg_content(text):
    start_marker = "start of the project gutenberg"
    end_marker = "end of the project gutenberg"

    start_idx = text.lower().find(start_marker)
    end_idx = text.lower().find(end_marker)

    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        return ""  # Skip files that don't have clear boundaries

    return text[start_idx + len(start_marker):end_idx].strip()


# Step 1: Load and preprocess text from multiple files
def load_text_files(data_dir):
    all_text = ""
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                raw_text = f.read()
                extracted_text = extract_gutenberg_content(raw_text)
                if extracted_text:
                    all_text += clean_text(extracted_text) + " "  # Add space to separate books
    return all_text


print("Loading and preprocessing text files...")
corpus = load_text_files(DATA_DIR)

# Step 2: Split text into words
words = corpus.split()


# Step 3: Generate prompt-completion pairs with variable-length prompts
def generate_pairs(words, min_seq_length, max_seq_length):
    pairs = []
    i = 0
    while i < len(words) - max_seq_length:
        seq_length = random.randint(min_seq_length, max_seq_length)
        if i + seq_length < len(words):
            prompt = " ".join(words[i : i + seq_length])
            if prompt[0].isupper():
                prompt = '<bos>' + prompt
            completion = words[i + seq_length]  # Predict the next word
            if completion.endswith('.') or completion.endswith('?') or completion.endswith('!'):
                completion += '<eos>'
            pairs.append({"prompt": prompt, "completion": completion})
        i += seq_length  # Move forward by the chosen sequence length
    return pairs

from typing import Tuple
def add_special_tokens(pairs: Tuple[list]):
    """
    Insert <bos> and <eos> special tokens into a dataset
    :param pairs: original prompts and completions
    :return: prompts/completion pairs with special tokens inserted
    """
    new_prompts = []
    new_completions = []

    for prompt, completion in zip(pairs):
        # If the beginning of the prompt is upper case, then we assume it is the start of a sequence
        if prompt[0].isupper():
            prompt = '<bos>' + prompt
        # If the end of the completion is a terminating punctuation, then we assume it is the end of a sequence
        if completion.endswith('.') or completion.endswith('?') or completion.endswith('!'):
            completion += '<eos>'
        new_prompts.append(prompt)
        new_completions.append(completion)

    return new_prompts, new_completions

print("Creating prompt-completion pairs with variable lengths...")
pairs = generate_pairs(words, MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)

# **SHUFFLE before splitting**
random.shuffle(pairs)

# Step 4: Split into training and testing sets
split_idx = int(len(pairs) * TRAIN_RATIO)
train_pairs, test_pairs = pairs[:split_idx], pairs[split_idx:]




# Step 5: Save datasets as JSONL files
def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


print("Saving training and testing datasets...")
save_jsonl(train_pairs, os.path.join(OUTPUT_DIR, "train.jsonl"))
save_jsonl(test_pairs, os.path.join(OUTPUT_DIR, "test.jsonl"))

# Save metadata
metadata = {
    "train_ratio": TRAIN_RATIO,
    "num_train_examples": len(train_pairs),
    "num_test_examples": len(test_pairs),
    "min_sequence_length": MIN_SEQ_LENGTH,
    "max_sequence_length": MAX_SEQ_LENGTH
}
with open(os.path.join(OUTPUT_DIR, "dataset_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print("Dataset preparation complete!")
