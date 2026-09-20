import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse"
).to(device)

vae.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Random test image
image = torch.randn(
    1, 3, 128, 128,
    device=device
).clamp(-1, 1)

with torch.no_grad():
    latent = vae.encode(image).latent_dist.sample()
    reconstructed = vae.decode(latent / vae.config.scaling_factor).sample

print("Image:       ", image.shape)
print("Latent:      ", latent.shape)
print("Reconstructed:", reconstructed.shape)