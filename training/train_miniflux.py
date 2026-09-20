import torch
from torch.utils.data import DataLoader

from datasets import load_from_disk
from torchvision import transforms

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
# SETTINGS
# ============================================================

BATCH_SIZE = 2
EPOCHS =20
LR = 1e-4

IMAGE_SIZE = 128


# ============================================================
# DATA
# ============================================================

dataset = load_from_disk(
    "../data/coco"
)


if "train" in dataset:

    dataset = dataset["train"]


# Keep the first small subset for Mac development
dataset = dataset.select(
    range(
        min(800, len(dataset))
    )
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


def prepare_image(example):

    image = example["image"].convert(
        "RGB"
    )

    image = transform(
        image
    )

    return {
        "pixel_values": image,
        "caption": example["text"]
    }


dataset = dataset.map(
    prepare_image
)

dataset.set_format(
    type="torch",
    columns=["pixel_values"],
    output_all_columns=True
)

# ============================================================
# VAE
# ============================================================

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse"
).to(device)

vae.eval()

for p in vae.parameters():
    p.requires_grad = False


# ============================================================
# TEXT ENCODER
# ============================================================

text_encoder = TextEncoder(
    device
)


# ============================================================
# MINI-FLUX
# ============================================================

model = MiniFlux(
    hidden_dim=256,
    double_blocks=2,
    single_blocks=4,
    heads=4
).to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

model.train()


for epoch in range(EPOCHS):

    total_loss = 0.0

    for index in range(
        0,
        len(dataset),
        BATCH_SIZE
    ):

        batch = dataset[
            index:index + BATCH_SIZE
        ]

        images = batch["pixel_values"].to(device)

        captions = batch["caption"]


        # ----------------------------------------------------
        # IMAGE → LATENT
        # ----------------------------------------------------

        with torch.no_grad():

            latents = vae.encode(
                images
            ).latent_dist.sample()

            latents = (
                latents
                *
                vae.config.scaling_factor
            )


        # ----------------------------------------------------
        # TEXT → TOKENS
        # ----------------------------------------------------

        with torch.no_grad():

            text_tokens = (
                text_encoder.encode(
                    captions
                )
            )


        # ----------------------------------------------------
        # FLOW MATCHING
        # ----------------------------------------------------

        x0 = torch.randn_like(
            latents
        )

        t = torch.rand(
            latents.shape[0],
            device=device
        )

        t_view = t.view(
            -1,
            1,
            1,
            1
        )

        xt = (
            (1 - t_view) * x0
            +
            t_view * latents
        )

        target = (
            latents - x0
        )


        # ----------------------------------------------------
        # MINI-FLUX
        # ----------------------------------------------------

        prediction = model(
            xt,
            text_tokens,
            t
        )


        loss = torch.nn.functional.mse_loss(
            prediction,
            target
        )


        # ----------------------------------------------------
        # BACKPROP
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()


        total_loss += loss.item()


        if index % 50 == 0:

            print(
                f"Epoch {epoch+1}/{EPOCHS} "
                f"Step {index}/{len(dataset)} "
                f"Loss {loss.item():.5f}"
            )


    average = (
        total_loss
        /
        (len(dataset) / BATCH_SIZE)
    )

    print(
        f"\nEpoch {epoch+1} "
        f"Average Loss: {average:.5f}\n"
    )

    if (epoch + 1) % 5 == 0:
        torch.save(
            model.state_dict(),
            f"checkpoints/miniflux_epoch_{epoch+1}.pt"
        )

        print(
            f"Checkpoint saved: epoch {epoch+1}"
        )


# ============================================================
# SAVE
# ============================================================

torch.save(
    model.state_dict(),
    "../checkpoints/miniflux.pt"
)

print("MODEL SAVED 🔥")