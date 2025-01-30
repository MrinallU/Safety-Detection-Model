import torch
from torchvision import transforms
from PIL import Image
import os
from vae import VAE


def save_images(tensor_list, output_folder="decoded_images", vae=None):
    """
    Decodes a list of 1D tensors (size 32) into 3x224x224 tensors,
    converts them to images, and saves them into `output_folder`.
    """
    # Make sure the folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # A transform that converts a torch.Tensor [C, H, W] in [0,1] range to a PIL Image
    to_pil = transforms.ToPILImage()
    index = 0
    for idx, t in enumerate(tensor_list):
        # 1) Decode the tensor (size 32) into a (3, 224, 224) tensor
        decoded_tensor = vae.decode(t)  # shape [3, 224, 224]
        print(decoded_tensor.shape)
        # 2) Make sure values are in a valid range for an image (e.g., 0..1 or 0..255)
        decoded_tensor = decoded_tensor.clamp(0, 1)

        for tensor in decoded_tensor:
            img = to_pil(tensor)
            img_filename = f"decoded_image_{index}.png"
            img_path = os.path.join(output_folder, img_filename)
            img.save(img_path)
            index += 1

        print(f"Saved {img_path}")


def reconstruct(file="name"):
    reconstructed_tensors = []
    with open(f"{file}.txt", "r") as f:
        for line in f:
            data_str = line.strip()
            # Evaluate the Python list string (use caution if string content is from an untrusted source)
            data_list = eval(data_str)

            # Recreate the tensor
            t = torch.tensor(data_list)
            reconstructed_tensors.append(t)
    return reconstructed_tensors


device = "cpu"
model = VAE(latent_size=32).to(device)
checkpoint = torch.load("vae_weights.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

val_tensors = reconstruct("val_tensors")
pred_tensors = reconstruct("train_tensors")

save_images(val_tensors, "actual_images", model)
save_images(pred_tensors, "pred_images", model)
