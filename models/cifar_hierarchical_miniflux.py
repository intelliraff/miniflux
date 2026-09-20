"""A compact multiscale flow transformer for 32x32 RGB images.

Convolutions only change resolution and fuse the skip connection. Transformer
blocks remain the learned velocity backbone, operating at 16x16 and 8x8.
"""
import torch
from torch import nn

from models.mini_flux import TimeEmbedding
from models.mini_flux_v2 import ModulatedBlock, sinusoidal_positions


def _spatial_positions(height, width, dim, device, dtype):
    if dim % 4:
        raise ValueError("embedding dimension must be divisible by four")
    rows, cols = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    positions = torch.cat(
        (
            sinusoidal_positions(rows.flatten(), dim // 2),
            sinusoidal_positions(cols.flatten(), dim // 2),
        ),
        dim=-1,
    )
    return positions.to(dtype)


def _tokens(feature_map):
    return feature_map.flatten(2).transpose(1, 2)


def _feature_map(tokens, height, width):
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)


class HierarchicalCifarMiniFlux(nn.Module):
    """Two-resolution DiT with a local stem and U-shaped feature path."""

    def __init__(
        self,
        base_dim=128,
        bottleneck_dim=256,
        encoder_depth=2,
        bottleneck_depth=4,
        decoder_depth=2,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()
        if base_dim % 4 or bottleneck_dim % 4:
            raise ValueError("model dimensions must be divisible by four")
        self.base_dim = base_dim
        self.bottleneck_dim = bottleneck_dim

        # The local operators preserve edges while attention models long-range structure.
        self.stem = nn.Conv2d(3, base_dim, kernel_size=3, stride=2, padding=1)
        self.downsample = nn.Conv2d(base_dim, bottleneck_dim, kernel_size=3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose2d(bottleneck_dim, base_dim, kernel_size=4, stride=2, padding=1)
        self.skip_fusion = nn.Conv2d(2 * base_dim, base_dim, kernel_size=1)

        self.time_embedding = TimeEmbedding(bottleneck_dim)
        self.base_condition = nn.Linear(bottleneck_dim, base_dim)
        self.encoder = nn.ModuleList([ModulatedBlock(base_dim, heads) for _ in range(encoder_depth)])
        self.bottleneck = nn.ModuleList(
            [ModulatedBlock(bottleneck_dim, heads) for _ in range(bottleneck_depth)]
        )
        self.decoder = nn.ModuleList([ModulatedBlock(base_dim, heads) for _ in range(decoder_depth)])
        for block in (*self.encoder, *self.bottleneck, *self.decoder):
            block.mlp = nn.Sequential(
                block.mlp[0], block.mlp[1], nn.Dropout(dropout), block.mlp[2]
            )

        self.final_norm = nn.LayerNorm(base_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(base_dim, 2 * base_dim))
        self.output = nn.ConvTranspose2d(base_dim, 3, kernel_size=4, stride=2, padding=1)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, timesteps, extra=None):
        del extra
        batch, channels, height, width = x.shape
        if channels != 3 or height % 4 or width % 4:
            raise ValueError("requires RGB dimensions divisible by four")

        condition = self.time_embedding(timesteps * 1000.0)
        base_condition = self.base_condition(condition)

        half_h, half_w = height // 2, width // 2
        encoded = self.stem(x)
        tokens = _tokens(encoded) + _spatial_positions(
            half_h, half_w, self.base_dim, x.device, encoded.dtype
        )
        for block in self.encoder:
            tokens = block(tokens, base_condition)
        encoded = _feature_map(tokens, half_h, half_w)

        quarter_h, quarter_w = height // 4, width // 4
        low = self.downsample(encoded)
        tokens = _tokens(low) + _spatial_positions(
            quarter_h, quarter_w, self.bottleneck_dim, x.device, low.dtype
        )
        for block in self.bottleneck:
            tokens = block(tokens, condition)
        low = _feature_map(tokens, quarter_h, quarter_w)

        decoded = self.upsample(low)
        decoded = self.skip_fusion(torch.cat((decoded, encoded), dim=1))
        tokens = _tokens(decoded) + _spatial_positions(
            half_h, half_w, self.base_dim, x.device, decoded.dtype
        )
        for block in self.decoder:
            tokens = block(tokens, base_condition)
        shift, scale = self.final_modulation(base_condition).unsqueeze(1).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale) + shift
        return self.output(_feature_map(tokens, half_h, half_w))
