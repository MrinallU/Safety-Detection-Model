import torch
import pandas as pd
import torch.optim as optim
import numpy as np
from vae import VAE
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image


class LSTM(nn.Module):
    def __init__(
        self,
        num_classes=1,
        in_features=32,
        lstm_units=256,
        num_lstm_layers=1,
        bidirectional=False,
    ):
        super(LSTM, self).__init__()

        # 2. LSTM layers
        # We stack two LSTMs similar to your Keras example
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # The final output size of LSTM is lstm_units * 2 if bidirectional, else lstm_units
        lstm_output_size = lstm_units * (2 if bidirectional else 1)

        # 3. Dense layer for classification
        self.dense = nn.Linear(lstm_output_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)  # (batch, seq_len, lstm_output_size)

        # (C) Dense (Fully Connected) + Softmax:
        # want only the last time step:
        logits = self.dense(x)  # shape: (batch, seq_len, num_classes)

        # If you need probabilities per timestep:
        probabilities = F.sigmoid(logits)

        return probabilities

    def train_one_epoch(self, data, optimizer, device, seq_len=32):
        """
        :param data: list of dictionaries from load_data()
        :param optimizer: PyTorch optimizer
        :param device: 'cuda' or 'cpu'
        :param batch_size: how many samples per update step
        :return: average loss over the epoch
        """
        self.train()
        criterion = nn.BCELoss(reduction="sum")
        running_loss = 0.0

        # Shuffle data if desired
        # import random
        # random.shuffle(data)

        # Process the data
        for i in range(0, len(data), 1):
            if i + seq_len == len(data):
                break
            batch = data[i : i + seq_len]

            embeddings = [
                item["embedding"] for item in batch
            ]  # each is shape [1, latent_size]
            labels = [item["label"] for item in batch]  # each is 0 or 1

            # Concatenate embeddings along dim=0 => shape: [seq_len, latent_size]
            # (assuming each embedding is shape [1, 32])
            embeddings = torch.cat(embeddings, dim=0).unsqueeze(
                0
            )  # now shape [batch_size=1,seq_len, 32]

            # Convert labels to a tensor of shape [batch = 1,seq_len, 1]
            labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).unsqueeze(0)

            # Forward pass
            outputs = self.forward(embeddings)  # shape [seq_len, 1]
            last_time_step = outputs[-1, -1, 0]

            optimizer.zero_grad()
            loss = criterion(last_time_step, labels[0, -1, 0])
            # print(f"loss {loss}, {last_time_step}, {labels[0, -1, 0]}")
            # print(labels[0, -1, 0])

            # Backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Average loss per sample
        # If reduction='sum', running_loss is the sum over all samples
        epoch_loss = running_loss / len(data)

        return epoch_loss

    def train_model(self, data, device="cpu", epochs=20, lr=1e-3):
        """
        Main training loop.

        :param device: torch.device (e.g., 'cuda' or 'cpu')
        :param train_loader: a DataLoader of (images, labels)
        :param epochs: number of epochs to train
        :param lr: learning rate
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data, optimizer, device)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

        torch.save(self.state_dict(), "lstm_weights.pth")


def load_image(filepath):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = Image.open(filepath).convert("RGB")
    img_tensor = transform(img)
    return img_tensor  # shape [C, H, W]


def load_data(
    csv_path="safety_detection_labeled_data/Safety_Detection_Labeled.csv",
    images_folder="safety_detection_labeled_data/",
    vae_weights="vae_weights.pth",
):
    df = pd.read_csv(csv_path)

    model = VAE(latent_size=32)
    checkpoint = torch.load(vae_weights, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()

    data = []

    for _, row in df.iterrows():
        filename = row["Filename"]
        label = row["Label"]

        # Build full path to image
        img_path = os.path.join(images_folder, filename)

        # Make sure the file exists before loading
        if not os.path.isfile(img_path):
            print(f"Warning: {img_path} does not exist. Skipping.")
            continue

        # Load and process the image
        x = load_image(img_path)  # shape [C, H, W]
        # Reshape to add batch dimension => shape [1, C, H, W]
        x = x.unsqueeze(0)

        # Encode with the VAE
        with torch.no_grad():
            output, logvar = model.encode(x)
        data.append({
            "filename": filename,
            "embedding": output,  # or output_np
            "label": label,
        })
        # print(data)
    return data


def eval(
    csv_path="safety_detection_labeled_data/Safety_Detection_Labeled.csv",
    images_folder="safety_detection_labeled_data/",
    vae_weights="vae_weights.pth",
    lstm_weights="lstm_weights.pth",
    seq_len=32,
):
    data = load_data(
        csv_path=csv_path, images_folder=images_folder, vae_weights=vae_weights
    )
    model = LSTM()
    checkpoint = torch.load(lstm_weights, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()

    for i in range(0, len(data), 1):
        if i + seq_len == len(data):
            break
        batch = data[i : i + seq_len]

        embeddings = [item["embedding"] for item in batch]
        labels = [item["label"] for item in batch]

        embeddings = torch.cat(embeddings, dim=0).unsqueeze(0)
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).unsqueeze(0)

        # Forward pass
        outputs = model.forward(embeddings)
        last_time_step = outputs[-1, -1, 0]

        crit = nn.BCELoss()
        loss = crit(last_time_step, labels[0, -1, 0])
        print(f"Loss: {loss}, Output: {last_time_step}, Label: {labels[0, -1, 0]}")


if __name__ == "__main__":
    eval()