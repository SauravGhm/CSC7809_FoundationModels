from flask import Flask, request, jsonify, render_template
import pandas as pd
import torch
from torchvision import datasets, transforms
import os

app = Flask(__name__)

# Load Fashion MNIST test dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False)

# Get true labels
test_images, test_labels = next(iter(test_loader))
test_labels = test_labels.numpy()

# Store results
if not os.path.exists("results.csv"):
    df = pd.DataFrame(columns=["name", "accuracy"])
    df.to_csv("results.csv", index=False)

@app.route("/", methods=["GET"])
def leaderboard():
    """Display the leaderboard"""
    df = pd.read_csv("results.csv")
    df = df.sort_values(by="accuracy", ascending=False)
    return df.to_html(index=False)

@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json()

    if "name" not in data or "predictions" not in data:
        return jsonify({"error": "Invalid data format. Must include 'name' and 'predictions'"}), 400

    name = data["name"]
    predictions = data["predictions"]

    if len(predictions) != len(test_labels):
        return jsonify({"error": "Prediction length does not match test set"}), 400

    predictions = torch.tensor(predictions)
    accuracy = (predictions.numpy() == test_labels).mean() * 100

    # Read current results
    df = pd.read_csv("results.csv")

    # 1. Drop old rows for this name
    df = df[df["name"] != name]

    # 2. Append the new submission
    new_entry = pd.DataFrame([{"name": name, "accuracy": accuracy}])
    df = pd.concat([df, new_entry], ignore_index=True)

    # 3. Write back to CSV
    df.to_csv("results.csv", index=False)

    return jsonify({"message": f"Submission received. Accuracy: {accuracy:.2f}%"}), 200


if __name__ == "__main__":
    app.run(debug=True)
