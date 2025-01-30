import torch
import time
import pandas as pd
import torch.optim as optim
from vae import VAE
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import os
from PIL import Image
from sklearn.metrics import mean_squared_error
from PIL import Image
from torchvision.transforms import ToPILImage


class LSTM(nn.Module):
    def __init__(
        self,
        in_features=32,
        lstm_units=256,
        num_lstm_layers=1,
        bidirectional=False,
    ):
        super(LSTM, self).__init__()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_size = lstm_units * (2 if bidirectional else 1)

        self.dense = nn.Linear(lstm_output_size, 32)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dense(x)
        return x

    def train_one_epoch(self, data, optimizer, device, seq_len=32, horizon=10):
        self.train()
        criterion = nn.MSELoss(reduction="sum")
        running_loss = 0.0

        for i in range(0, len(data), 1):
            if i + seq_len + horizon >= len(data):
                break

            batch = data[i : i + seq_len + horizon]

            embeddings = [
                item["embedding"].to(device) for item in batch[0 : len(batch) - horizon]
            ]
            future_embeddings = [
                batch[j + horizon]["embedding"].to(device)
                for j in range(len(batch) - horizon)
            ]

            embeddings = torch.cat(embeddings, dim=0).unsqueeze(0).to(device)
            future_embeddings = (
                torch.cat(future_embeddings, dim=0).unsqueeze(0).to(device)
            )

            outputs = self.forward(embeddings)

            optimizer.zero_grad()
            loss = criterion(outputs, future_embeddings)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(data)
        return epoch_loss

    def train_model(
        self, data, device="cpu", seq_len=32, horizon=10, epochs=30, lr=1e-3
    ):
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        data = data[0:4000]
        finLoss = None

        for epoch in range(epochs):
            epoch_loss = self.train_one_epoch(data, optimizer, device, seq_len, horizon)
            if finLoss is None or finLoss > epoch_loss:
                finLoss = epoch_loss
                torch.save(self.state_dict(), f"lstm_weights{horizon}.pth")
                torch.save(self.state_dict(), "lstm_weights_pred.pth")
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

        print(f"Loss: {finLoss:.4f}")


def load_image(filepath, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = Image.open(filepath).convert("RGB")
    img_tensor = transform(img).to(device)
    return img_tensor


def load_data(
    csv_path="../safety_detection_labeled_data/Safety_Detection_Labeled.csv",
    images_folder="../safety_detection_labeled_data/",
    vae_weights="vae_weights.pth",
    device="cpu",
):
    df = pd.read_csv(csv_path)

    model = VAE(latent_size=32).to(device)
    checkpoint = torch.load(vae_weights, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    data = []

    for _, row in df.iterrows():
        filename = row["Filename"]
        label = row["Label"]
        img_path = os.path.join(images_folder, filename)

        if not os.path.isfile(img_path):
            print(f"Warning: {img_path} does not exist. Skipping.")
            continue

        x = load_image(img_path, device).unsqueeze(0)
        with torch.no_grad():
            output, logvar = model.encode(x)

        data.append({
            "filename": filename,
            "embedding": output.to(device),
            "label": label,
            "image": x.to(device),
        })

    return data


def eval(
    csv_path="../safety_detection_labeled_data/Safety_Detection_Labeled.csv",
    images_folder="../safety_detection_labeled_data/",
    vae_weights="vae_weights.pth",
    lstm_weights="lstm_weights_pred.pth",
    seq_len=32,
    horizon=10,
    load_lstm_weights=True,
    load_d=True,
    data=None,
    device="cpu",
):
    criterion = nn.MSELoss()
    if load_d:
        data = load_data(
            csv_path=csv_path,
            images_folder=images_folder,
            vae_weights=vae_weights,
            device=device,
        )

    data = data[4000:4100]
    model = LSTM().to(device)

    vae = VAE(latent_size=32).to(device)
    checkpoint = torch.load(vae_weights, map_location=device, weights_only=True)
    vae.load_state_dict(checkpoint)
    vae.eval()

    if load_lstm_weights:
        checkpoint = torch.load(lstm_weights, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    model.eval()
    to_pil = transforms.ToPILImage()

    all_preds = []
    all_outs = []
    index1 = 0
    index2 = 0

    for i in range(0, len(data), 1):
        if i + seq_len + horizon >= len(data):
            break

        batch = data[i : i + seq_len + horizon]
        embeddings = [item["embedding"] for item in batch[0 : len(batch) - horizon]]
        embeddings = torch.cat(embeddings, dim=0).unsqueeze(0).to(device)

        future_embeddings = [
            batch[j + horizon]["embedding"] for j in range(len(batch) - horizon)
        ]
        future_embeddings = torch.cat(future_embeddings, dim=0).unsqueeze(0).to(device)

        outputs = model.forward(embeddings)

        # SAVE IMAGES
        # decoded_outputs = vae.decode(outputs)
        # decoded_tensor = vae.decode(future_embeddings)  # shape [3, 224, 224]

        # for tensor in decoded_tensor:
        #     img = to_pil(tensor)
        #     img_filename = f"decoded_image_{index1}.png"
        #     img_path = os.path.join("actual_images", img_filename)
        #     img.save(img_path)
        #     index1 += 1

        # for tensor in decoded_outputs:
        #     img = to_pil(tensor)
        #     img_filename = f"decoded_image_{index2}.png"
        #     img_path = os.path.join("pred_images", img_filename)
        #     img.save(img_path)
        #     index2 += 1

        future_embeddings = future_embeddings[0]
        outputs = outputs[0]
        # print(f"{outputs.shape} {future_embeddings.shape}")

        all_preds.append(outputs)
        all_outs.append(future_embeddings)

    val_tensor = torch.stack(all_outs)
    model_tensor = torch.stack(all_preds)

    mse_val = criterion(val_tensor, model_tensor)

    print(f"MSE: {mse_val:.4f}")
    return mse_val


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lens = [32]
    horizon_init = 10
    horizon_increment = 5
    horizon_limit = 100
    # Training
    data = load_data(device=device)
    print("DATA loaded")
    model = LSTM()
    for h in range(horizon_init, horizon_limit + 1, horizon_increment):
        for l in lens:
            print(f"Results for Horizon {h} and Sequence Length {l}:")
            print("_______________________________________________")
            model.train_model(data=data, seq_len=l, horizon=h, device=device, epochs=10)
            mse = eval(
                load_lstm_weights=True,
                load_d=False,
                data=data,
                horizon=h,
                seq_len=l,
                device=device,
            )
            with open("results.txt", "a") as file:
                file.write(f"Results for Horizon {h} and Sequence Length {l}:\n")
                file.write("_______________________________________________\n")
                file.write(f"MSE: {mse: .4f}\n")
                # file.write(f"ssim: {ssim: .4f}\n")
