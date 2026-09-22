"""Class-conditioned local-global MiniFlux for CIFAR-10 flow matching."""
import torch
from torch import nn

from models.cifar_hybrid_miniflux import CifarHybridMiniFlux


class ConditionedCifarHybridMiniFlux(CifarHybridMiniFlux):
    """Hybrid MiniFlux with ten class embeddings and a learned null class.

    The timestep and class embeddings are added before entering the unchanged
    conditioned convolution and transformer blocks. The inherited architecture
    remains the same apart from the compact 11-entry embedding table.
    """

    NUM_CLASSES = 10
    NULL_CLASS = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.class_embedding = nn.Embedding(self.NUM_CLASSES + 1, self.bottleneck_dim)
        nn.init.normal_(self.class_embedding.weight, std=0.02)

    def forward(self, x, timesteps, labels=None, extra=None):
        del extra
        if labels is None:
            labels = torch.full(
                (x.shape[0],), self.NULL_CLASS, device=x.device, dtype=torch.long
            )
        else:
            labels = torch.as_tensor(labels, device=x.device, dtype=torch.long)
        if labels.shape != (x.shape[0],):
            raise ValueError("labels must have shape (batch,)")
        if ((labels < 0) | (labels > self.NULL_CLASS)).any():
            raise ValueError("labels must be CIFAR-10 classes 0-9 or null class 10")
        # Shared core expects a combined timestep condition. Adding the class
        # embedding here conditions every local residual and modulated block.
        condition = self.time_embedding(timesteps * 1000.0) + self.class_embedding(labels)
        return self.forward_conditioned(x, condition)
