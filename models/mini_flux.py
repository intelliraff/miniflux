import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TIME EMBEDDING
# ============================================================

class TimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):

        half = self.dim // 2

        freq = torch.exp(
            -math.log(10000)
            * torch.arange(
                half,
                device=t.device,
                dtype=torch.float32
            )
            / half
        )

        args = t[:, None] * freq[None, :]

        emb = torch.cat(
            [torch.sin(args), torch.cos(args)],
            dim=-1
        )

        return self.mlp(emb)


# ============================================================
# ROTARY EMBEDDING
# ============================================================

def rotate_half(x):

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    return torch.stack(
        [-x2, x1],
        dim=-1
    ).flatten(-2)


def apply_rope(q, k, positions):

    dim = q.shape[-1]

    half = dim // 2

    freq = torch.exp(
        -math.log(10000)
        * torch.arange(
            half,
            device=q.device,
            dtype=torch.float32
        )
        / half
    )

    angles = positions[..., None] * freq

    sin = torch.sin(angles)
    cos = torch.cos(angles)

    sin = torch.repeat_interleave(
        sin,
        2,
        dim=-1
    )

    cos = torch.repeat_interleave(
        cos,
        2,
        dim=-1
    )

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    return q, k


# ============================================================
# MULTIHEAD ATTENTION
# ============================================================

class Attention(nn.Module):

    def __init__(
        self,
        dim,
        heads=4
    ):

        super().__init__()

        self.heads = heads
        self.head_dim = dim // heads

        assert dim % heads == 0

        self.qkv = nn.Linear(
            dim,
            dim * 3
        )

        self.out = nn.Linear(
            dim,
            dim
        )

    def forward(
        self,
        x,
        use_rope=True
    ):

        B, N, D = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            B,
            N,
            3,
            self.heads,
            self.head_dim
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4
        )

        q, k, v = qkv

        if use_rope:

            positions = torch.arange(
                N,
                device=x.device,
                dtype=torch.float32
            )

            positions = positions.unsqueeze(0)

            q, k = apply_rope(
                q,
                k,
                positions
            )

        attention = F.scaled_dot_product_attention(
            q,
            k,
            v
        )

        attention = attention.transpose(
            1,
            2
        )

        attention = attention.reshape(
            B,
            N,
            D
        )

        return self.out(attention)


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        dim,
        expansion=4
    ):

        super().__init__()

        hidden = dim * expansion

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# DOUBLE STREAM BLOCK
#
# FLUX INSPIRED:
#
# text stream
# image stream
#      ↓
# joint attention
# ============================================================

class DoubleStreamBlock(nn.Module):

    def __init__(
        self,
        dim,
        heads=4
    ):

        super().__init__()

        self.text_norm = nn.LayerNorm(dim)
        self.image_norm = nn.LayerNorm(dim)

        self.text_attention = Attention(
            dim,
            heads
        )

        self.image_attention = Attention(
            dim,
            heads
        )

        self.joint_attention = Attention(
            dim,
            heads
        )

        self.text_mlp = MLP(dim)
        self.image_mlp = MLP(dim)

        self.time_text = nn.Linear(
            dim,
            dim
        )

        self.time_image = nn.Linear(
            dim,
            dim
        )

    def forward(
        self,
        text,
        image,
        conditioning
    ):

        # -----------------------------------------------
        # Conditioning
        # -----------------------------------------------

        text = text + self.time_text(
            conditioning
        ).unsqueeze(1)

        image = image + self.time_image(
            conditioning
        ).unsqueeze(1)

        # -----------------------------------------------
        # Separate normalization
        # -----------------------------------------------

        text_norm = self.text_norm(text)
        image_norm = self.image_norm(image)

        # -----------------------------------------------
        # Joint attention
        # -----------------------------------------------

        combined = torch.cat(
            [text_norm, image_norm],
            dim=1
        )

        combined = self.joint_attention(
            combined
        )

        text_len = text.shape[1]

        text_attention = combined[:, :text_len]
        image_attention = combined[:, text_len:]

        # -----------------------------------------------
        # Residual
        # -----------------------------------------------

        text = text + text_attention
        image = image + image_attention

        # -----------------------------------------------
        # MLP
        # -----------------------------------------------

        text = text + self.text_mlp(
            text
        )

        image = image + self.image_mlp(
            image
        )

        return text, image


# ============================================================
# SINGLE STREAM BLOCK
# ============================================================

class SingleStreamBlock(nn.Module):

    def __init__(
        self,
        dim,
        heads=4
    ):

        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.attention = Attention(
            dim,
            heads
        )

        self.mlp = MLP(dim)

    def forward(self, x):

        h = self.norm(x)

        x = x + self.attention(h)

        x = x + self.mlp(
            self.norm(x)
        )

        return x


# ============================================================
# IMAGE PATCH EMBEDDING
# ============================================================

class ImageTokenizer(nn.Module):

    def __init__(
        self,
        channels=4,
        patch_size=2,
        dim=256
    ):

        super().__init__()

        self.patch_size = patch_size

        self.projection = nn.Conv2d(
            channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):

        x = self.projection(x)

        x = x.flatten(2)

        x = x.transpose(
            1,
            2
        )

        return x


# ============================================================
# TEXT PROJECTION
# ============================================================

class TextProjection(nn.Module):

    def __init__(
        self,
        input_dim=512,
        dim=256
    ):

        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            dim
        )

    def forward(self, text):

        return self.projection(
            text
        )


# ============================================================
# MINI-FLUX
# ============================================================

class MiniFlux(nn.Module):

    def __init__(
        self,

        latent_channels=4,

        text_dim=512,

        hidden_dim=256,

        double_blocks=2,

        single_blocks=4,

        heads=4,

        time_scale=1.0
    ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.time_scale = time_scale

        # -----------------------------------------------
        # Image tokens
        # -----------------------------------------------

        self.image_tokenizer = ImageTokenizer(
            channels=latent_channels,
            patch_size=2,
            dim=hidden_dim
        )

        # -----------------------------------------------
        # Text tokens
        # -----------------------------------------------

        self.text_projection = TextProjection(
            text_dim,
            hidden_dim
        )

        # -----------------------------------------------
        # Time
        # -----------------------------------------------

        self.time_embedding = TimeEmbedding(
            hidden_dim
        )

        # -----------------------------------------------
        # Double stream
        # -----------------------------------------------

        self.double_blocks = nn.ModuleList([

            DoubleStreamBlock(
                hidden_dim,
                heads
            )

            for _ in range(double_blocks)

        ])

        # -----------------------------------------------
        # Single stream
        # -----------------------------------------------

        self.single_blocks = nn.ModuleList([

            SingleStreamBlock(
                hidden_dim,
                heads
            )

            for _ in range(single_blocks)

        ])

        # -----------------------------------------------
        # Final output
        # -----------------------------------------------

        self.final_norm = nn.LayerNorm(
            hidden_dim
        )

        self.final = nn.Linear(
            hidden_dim,
            4 * 2 * 2
        )

    def forward(
        self,
        latent,
        text,
        timestep
    ):

        # ===============================================
        # IMAGE → IMAGE TOKENS
        # ===============================================

        image = self.image_tokenizer(
            latent
        )

        # ===============================================
        # TEXT → TEXT TOKENS
        # ===============================================

        text = self.text_projection(
            text
        )

        # ===============================================
        # TIME
        # ===============================================

        conditioning = self.time_embedding(
            timestep * self.time_scale
        )

        # ===============================================
        # DOUBLE STREAM
        # ===============================================

        for block in self.double_blocks:

            text, image = block(
                text,
                image,
                conditioning
            )

        # ===============================================
        # SINGLE STREAM
        # ===============================================

        combined = torch.cat(
            [text, image],
            dim=1
        )

        for block in self.single_blocks:

            combined = block(
                combined
            )

        # ===============================================
        # KEEP IMAGE TOKENS
        # ===============================================

        image_start = text.shape[1]

        image = combined[
            :,
            image_start:
        ]

        # ===============================================
        # FINAL PROJECTION
        # ===============================================

        image = self.final_norm(
            image
        )

        image = self.final(
            image
        )

        # ===============================================
        # TOKENS → LATENT
        # ===============================================

        B = image.shape[0]

        H = latent.shape[2]
        W = latent.shape[3]

        patch = 2

        gh = H // patch
        gw = W // patch

        image = image.reshape(
            B,
            gh,
            gw,
            patch,
            patch,
            4
        )

        image = image.permute(
            0,
            5,
            1,
            3,
            2,
            4
        )

        image = image.reshape(
            B,
            4,
            H,
            W
        )

        return image


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = MiniFlux().to(device)

    latent = torch.randn(
        2,
        4,
        16,
        16,
        device=device
    )

    text = torch.randn(
        2,
        77,
        768,
        device=device
    )

    timestep = torch.rand(
        2,
        device=device
    )

    with torch.no_grad():

        output = model(
            latent,
            text,
            timestep
        )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Device:", device)
    print("Input latent:", latent.shape)
    print("Text:", text.shape)
    print("Output:", output.shape)
    print(
        "Parameters:",
        f"{parameters:,}"
    )
