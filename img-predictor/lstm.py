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

        # Decoder layers
        self.dec_fc = nn.Linear(lstm_output_size, 256 * 14 * 14)
        self.dec_conv1 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        )
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dec_fc(x)
        x = x.view(-1, 256, 14, 14)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = F.sigmoid(self.dec_conv4(x))
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
            images = [
                batch[j + horizon]["image"].to(device)
                for j in range(len(batch) - horizon)
            ]

            embeddings = torch.cat(embeddings, dim=0).unsqueeze(0).to(device)
            images = torch.cat(images, dim=0).unsqueeze(0).to(device)

            outputs = self.forward(embeddings)
            last_time_step = outputs[-1]

            optimizer.zero_grad()
            loss = criterion(last_time_step, images[0, -1])

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
    checkpoint = torch.load(vae_weights, map_location=device)
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
    criterion = nn.MSELoss(reduction="sum")
    if load_d:
        data = load_data(
            csv_path=csv_path,
            images_folder=images_folder,
            vae_weights=vae_weights,
            device=device,
        )

    data = data[1:50]
    model = LSTM().to(device)

    if load_lstm_weights:
        checkpoint = torch.load(lstm_weights, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    model.eval()

    output_folder = "saved_images"
    os.makedirs(output_folder, exist_ok=True)
    to_pil = ToPILImage()

    all_preds = []
    all_imgs = []
    for i in range(0, len(data), 1):
        if i + seq_len + horizon >= len(data):
            break

        batch = data[i : i + seq_len + horizon]
        embeddings = [item["embedding"] for item in batch[0 : len(batch) - horizon]]
        embeddings = torch.cat(embeddings, dim=0).unsqueeze(0).to(device)

        images = [batch[j + horizon]["image"] for j in range(len(batch) - horizon)]
        images = torch.cat(images, dim=0).unsqueeze(0).to(device)

        outputs = model.forward(embeddings)
        last_time_step = outputs[-1]

        true_image = images[-1, -1]
        pil_image = to_pil(true_image.cpu())  # Convert tensor to PIL Image
        image_path = os.path.join(output_folder, f"true_image_{i}.png")
        pil_image.save(image_path)

        pil_image = to_pil(last_time_step.cpu())  # Convert tensor to PIL Image
        image_path = os.path.join(output_folder, f"model_image_{i}.png")
        pil_image.save(image_path)

        all_preds.append(last_time_step)
        all_imgs.append(true_image)

    val_tensor = torch.stack(all_imgs)
    model_tensor = torch.stack(all_preds)
    mse_val = criterion(val_tensor, model_tensor)

    print(f"MSE: {mse_val:.4f}")
    return mse_val


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lens = [32]
    horizon_init = 0
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
            # model.train_model(data=data, seq_len=l, horizon=h, device=device)
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
