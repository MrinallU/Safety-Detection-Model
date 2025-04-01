import math
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
from matplotlib.widgets import Slider
import pandas as pd
from torchvision import transforms


# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, latent_size=32):
        super(VAE, self).__init__()
        self.latent_size = latent_size

        # ---------------------------
        #         Encoder
        # ---------------------------
        # Input shape: (3, 224, 224)
        # Downsample by factor of 2 with each conv layer:
        #   224 -> 112 -> 56 -> 28 -> 14
        self.enc_conv1 = nn.Conv2d(
            3, 32, kernel_size=4, stride=2, padding=1
        )  # (3, 224,224) -> (32, 112,112)
        self.enc_conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # (32,112,112) -> (64, 56,56)
        self.enc_conv3 = nn.Conv2d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # (64, 56,56)  -> (128,28,28)
        self.enc_conv4 = nn.Conv2d(
            128, 256, kernel_size=4, stride=2, padding=1
        )  # (128,28,28)  -> (256,14,14)

        # Now the flattened feature map is (256 * 14 * 14) = 50176
        self.fc_mu = nn.Linear(256 * 14 * 14, latent_size)
        self.fc_logvar = nn.Linear(256 * 14 * 14, latent_size)

        # ---------------------------
        #         Decoder
        # ---------------------------
        # We do the reverse: 14 -> 28 -> 56 -> 112 -> 224
        self.dec_fc = nn.Linear(latent_size, 256 * 14 * 14)

        self.dec_conv1 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1
        )  # (256,14,14) -> (128,28,28)
        self.dec_conv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # (128,28,28)-> (64,56,56)
        self.dec_conv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # (64,56,56) -> (32,112,112)
        self.dec_conv4 = nn.ConvTranspose2d(
            32, 3, kernel_size=4, stride=2, padding=1
        )  # (32,112,112)-> (3,224,224)

    def reparameterize_mean(self, mu, logvar):
        # Use mean value for reparameterization
        return mu

    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from N(mu, var) from N(0,1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        # Encode input image to latent space
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        # Decode latent space to output image
        x = F.relu(self.dec_fc(z))
        x = x.view(
            -1, 256, 14, 14
        )  # Reshape to match the beginning shape of the decoder
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = F.relu(self.dec_conv4(x))
        return x

    def train_step(self, x, optimizer, label):
        # Set the model to training mode
        self.train()
        # Forward pass
        recon_x, mu, logvar = self.forward(x)
        # Compute loss
        loss, recon, kl = self.loss_function(recon_x, x, mu, logvar, label)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return [loss.item(), recon, kl]

    # Define the loss function for the VAE
    def loss_function(self, recon_x, x, mu, logvar, label):
        MSE = nn.MSELoss(reduction="sum")
        reconstruction_loss = MSE(recon_x, x)
        a = 3 if label == 0 else -3
        kl_divergence = -0.5 * torch.sum(1 + logvar - (mu - a).pow(2) - logvar.exp())
        return [reconstruction_loss + kl_divergence, reconstruction_loss, kl_divergence]

    def forward(self, x):
        # Forward pass through the VAE
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def forward_eval(self, x):
        # Forward pass through the VAE
        mu, logvar = self.encode(x)
        z = self.reparameterize_mean(mu, logvar)
        return self.decode(z)


def load_image(filepath):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = Image.open(filepath).convert("RGB")
    img_tensor = transform(img)
    return img_tensor  # shape [C, H, W]


def load_data(
    csv_path="./safety_detection_labeled_data/Safety_Detection_Labeled.csv",
    images_folder="./safety_detection_labeled_data/",
    device="cpu",
):
    df = pd.read_csv(csv_path)

    data = []

    for _, row in df.iterrows():
        filename = row["Filename"]
        label = row["Label"]
        img_path = os.path.join(images_folder, filename)
        if not os.path.isfile(img_path):
            print(f"Warning: {img_path} does not exist. Skipping.")
            continue
        x = load_image(img_path)  # shape [C, H, W]
        x = x.unsqueeze(0).to(device)

        data.append({
            "filename": filename,
            "image": x,
            "label": label,
        })
    data = sorted(
        data, key=lambda item: int(item["filename"].split("_")[1].split(".")[0])
    )
    return data


def train():
    image_array = load_data()
    image_array = image_array[0:4000]
    model = VAE(latent_size=32)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        avg_recon = 0
        avg_kl = 0
        n = 0
        for data_point in image_array:
            x = data_point["image"]
            loss, recon, kl = model.train_step(x, optimizer, data_point["label"])
            epoch_loss += loss
            avg_kl += kl
            avg_recon += recon
            n += 1
            # print(f"Loss: {loss}")

        avg_loss = epoch_loss / len(image_array)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, kl_divergence: {avg_kl / n} recon: {avg_recon / n}"
        )
        torch.save(model.state_dict(), "vae_weights_prior_shift_gap.pth")


def eval(x):
    model = VAE(latent_size=32)
    checkpoint = torch.load("vae_weights_split.pth", weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()

    output = model.forward_eval(x)
    output = output.squeeze(0)
    output = output.permute(1, 2, 0)  # H, W, C
    output = output.detach().numpy()
    output = output[..., ::-1]  # RGB to BGR
    return output


if __name__ == "__main__":
    # folder_path = "/home/mrinall/TEA/hsai-predictor/MonoLstm/version2/safety_detection_labeled_data"
    train()
    # image_array = load_data()
    # x = image_array[10]["image"]
    # print(x.shape)

    # out = eval(x)
    # cv2.imshow("Image", out)
    # cv2.waitKey(0)
