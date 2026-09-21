"""Local-global MiniFlux for 32x32 RGB flow matching.

Small residual convolution blocks learn edges and textures at 32x32 and 16x16.
Modulated transformer blocks model global structure at the 8x8 bottleneck.
"""
import torch
from torch import nn

from models.mini_flux import TimeEmbedding
from models.mini_flux_v2 import ModulatedBlock
from models.cifar_hierarchical_miniflux import (
    _feature_map,
    _spatial_positions,
    _tokens,
)


class ConditionedResidualBlock(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * channels))
        self.activation = nn.SiLU()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, condition):
        shift, scale = self.condition(condition).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm1(x) * (1 + scale) + shift
        h = self.conv1(self.activation(h))
        h = self.conv2(self.activation(self.norm2(h)))
        return x + h


class CifarHybridMiniFlux(nn.Module):
    """Convolutional detail path with a transformer-only global bottleneck."""

    def __init__(
        self,
        local_dim=64,
        base_dim=128,
        bottleneck_dim=256,
        bottleneck_depth=5,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()
        if local_dim % 8 or base_dim % 8 or bottleneck_dim % 4:
            raise ValueError("local dimensions need groups of eight; bottleneck needs groups of four")
        self.bottleneck_dim = bottleneck_dim
        self.input_projection = nn.Conv2d(3, local_dim, 3, padding=1)
        self.local_encoder = ConditionedResidualBlock(local_dim, bottleneck_dim)
        self.down_to_base = nn.Conv2d(local_dim, base_dim, 3, stride=2, padding=1)
        self.base_encoder = ConditionedResidualBlock(base_dim, bottleneck_dim)
        self.down_to_bottleneck = nn.Conv2d(base_dim, bottleneck_dim, 3, stride=2, padding=1)

        self.time_embedding = TimeEmbedding(bottleneck_dim)
        self.transformer = nn.ModuleList(
            [ModulatedBlock(bottleneck_dim, heads) for _ in range(bottleneck_depth)]
        )
        for block in self.transformer:
            block.mlp = nn.Sequential(
                block.mlp[0], block.mlp[1], nn.Dropout(dropout), block.mlp[2]
            )

        self.up_to_base = nn.ConvTranspose2d(
            bottleneck_dim, base_dim, kernel_size=4, stride=2, padding=1
        )
        self.base_fusion = nn.Conv2d(2 * base_dim, base_dim, 1)
        self.base_decoder = ConditionedResidualBlock(base_dim, bottleneck_dim)
        self.up_to_local = nn.ConvTranspose2d(
            base_dim, local_dim, kernel_size=4, stride=2, padding=1
        )
        self.local_fusion = nn.Conv2d(2 * local_dim, local_dim, 1)
        self.local_decoder = ConditionedResidualBlock(local_dim, bottleneck_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(8, local_dim),
            nn.SiLU(),
            nn.Conv2d(local_dim, 3, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x, timesteps, extra=None):
        del extra
        batch, channels, height, width = x.shape
        if channels != 3 or height % 4 or width % 4:
            raise ValueError("requires RGB dimensions divisible by four")
        condition = self.time_embedding(timesteps * 1000.0)

        local = self.local_encoder(self.input_projection(x), condition)
        base = self.base_encoder(self.down_to_base(local), condition)
        low = self.down_to_bottleneck(base)
        low_h, low_w = height // 4, width // 4
        tokens = _tokens(low) + _spatial_positions(
            low_h, low_w, self.bottleneck_dim, x.device, low.dtype
        )
        for block in self.transformer:
            tokens = block(tokens, condition)
        low = _feature_map(tokens, low_h, low_w)

        decoded_base = self.up_to_base(low)
        decoded_base = self.base_fusion(torch.cat((decoded_base, base), dim=1))
        decoded_base = self.base_decoder(decoded_base, condition)
        decoded_local = self.up_to_local(decoded_base)
        decoded_local = self.local_fusion(torch.cat((decoded_local, local), dim=1))
        decoded_local = self.local_decoder(decoded_local, condition)
        return self.output(decoded_local)
