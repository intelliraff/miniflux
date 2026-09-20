import torch
import torch.nn.functional as F


def sample_flow_path(
    x0,
    x1,
    t
):

    t = t.view(
        -1,
        1,
        1,
        1
    )

    xt = (
        (1 - t) * x0
        +
        t * x1
    )

    velocity = x1 - x0

    return xt, velocity


def flow_matching_loss(
    model,
    x1,
    text,
    labels=None
):

    batch_size = x1.shape[0]

    x0 = torch.randn_like(
        x1
    )

    t = torch.rand(
        batch_size,
        device=x1.device
    )

    xt, target_velocity = sample_flow_path(
        x0,
        x1,
        t
    )

    predicted_velocity = model(
        xt,
        text,
        t
    )

    loss = F.mse_loss(
        predicted_velocity,
        target_velocity
    )

    return loss