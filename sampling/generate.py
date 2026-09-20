import os
import torch
import matplotlib.pyplot as plt

from diffusers import AutoencoderKL

from models.mini_flux import MiniFlux
from models.text_encoder import TextEncoder


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 128

# SD VAE with 128x128 images -> 16x16 latent
LATENT_SIZE = 16

# Number of Flow Matching integration steps
STEPS = 30

# Use the 20-epoch checkpoint
CHECKPOINT = "checkpoints/miniflux_epoch_20.pt"


# ============================================================
# PROMPTS
# ============================================================

PROMPTS = [
    "a dog sitting on grass",
    "a red car on a road",
    "a cat sitting on a chair",
    "a person riding a bicycle"
]


# ============================================================
# LOAD VAE
# ============================================================

print()
print("Loading VAE...")

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse"
).to(device)

vae.eval()

for parameter in vae.parameters():
    parameter.requires_grad = False

print("VAE loaded!")


# ============================================================
# LOAD TEXT ENCODER
# ============================================================

print()
print("Loading CLIP...")

text_encoder = TextEncoder(
    device
)

print("CLIP loaded!")


# ============================================================
# LOAD MINI-FLUX
# ============================================================

print()
print("Loading Mini-FLUX...")

model = MiniFlux(
    latent_channels=4,
    text_dim=512,
    hidden_dim=256,
    double_blocks=2,
    single_blocks=4,
    heads=4
).to(device)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    checkpoint
)

model.eval()

print("Mini-FLUX loaded!")
print("Checkpoint:", CHECKPOINT)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


# ============================================================
# SAME INITIAL NOISE FOR EVERY PROMPT
# ============================================================

print()
print("Creating shared initial noise...")

torch.manual_seed(42)

if device.type == "mps":
    torch.mps.manual_seed(42)

initial_noise = torch.randn(
    1,
    4,
    LATENT_SIZE,
    LATENT_SIZE,
    device=device
)

print(
    "Initial noise shape:",
    initial_noise.shape
)


# ============================================================
# GENERATE ONE IMAGE FOR EACH PROMPT
# ============================================================

for prompt_index, prompt in enumerate(PROMPTS):

    print()
    print("=" * 60)
    print(
        f"PROMPT {prompt_index + 1}/{len(PROMPTS)}"
    )
    print(prompt)
    print("=" * 60)


    # --------------------------------------------------------
    # RESET TO EXACT SAME INITIAL NOISE
    # --------------------------------------------------------

    x = initial_noise.clone()


    # --------------------------------------------------------
    # TEXT → CLIP EMBEDDING
    # --------------------------------------------------------

    print("Encoding text...")

    with torch.no_grad():

        text = text_encoder.encode(
            [prompt]
        )

    print(
        "Text shape:",
        text.shape
    )


    # --------------------------------------------------------
    # FLOW MATCHING ODE
    # --------------------------------------------------------

    print("Generating...")

    with torch.no_grad():

        for step in range(STEPS):

            # Time goes from 0 → 1
            t_value = step / STEPS

            t = torch.tensor(
                [t_value],
                device=device,
                dtype=torch.float32
            )

            # Predict velocity
            velocity = model(
                x,
                text,
                t
            )

            # Euler integration
            dt = 1.0 / STEPS

            x = x + velocity * dt

            if step % 5 == 0:

                print(
                    f"Step {step}/{STEPS}"
                )


    # --------------------------------------------------------
    # LATENT → IMAGE
    # --------------------------------------------------------

    print("Decoding image...")

    with torch.no_grad():

        decoded = x / vae.config.scaling_factor

        image = vae.decode(
            decoded
        ).sample


    # --------------------------------------------------------
    # NORMALIZE IMAGE
    # --------------------------------------------------------

    image = (
        image.clamp(-1, 1) + 1
    ) / 2

    image = (
        image[0]
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )


    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    safe_name = (
        prompt
        .replace(" ", "_")
        .replace("/", "_")
    )

    output_path = (
        f"outputs/"
        f"diagnostic_{prompt_index + 1}_"
        f"{safe_name}.png"
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    plt.figure(
        figsize=(5, 5)
    )

    plt.imshow(
        image
    )

    plt.title(
        f"Mini-FLUX\n{prompt}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.show()


    print()
    print(
        "Saved:",
        output_path
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("TEXT-CONDITIONING DIAGNOSTIC COMPLETE")
print("=" * 60)

print()
print("Generated images:")

for prompt_index, prompt in enumerate(PROMPTS):

    safe_name = (
        prompt
        .replace(" ", "_")
        .replace("/", "_")
    )

    print(
        f"{prompt_index + 1}. "
        f"outputs/diagnostic_{prompt_index + 1}_"
        f"{safe_name}.png"
    )

print()
print("IMPORTANT:")
print("All four prompts used the SAME initial noise.")
print("Only the text prompt was changed.")