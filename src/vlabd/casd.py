from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def average_feature_shift(
    base_model: nn.Module,
    changed_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    base_model.eval()
    changed_model.eval()
    sums: dict[str, torch.Tensor] = {}
    count = 0

    for batch in loader:
        image = batch["image"].to(device)
        tokens = batch["tokens"].to(device)
        _, base_feats = base_model(image, tokens, return_features=True)
        _, changed_feats = changed_model(image, tokens, return_features=True)

        bs = image.shape[0]
        count += bs
        for name in base_feats:
            diff = (changed_feats[name] - base_feats[name]).abs().mean(dim=0).detach().cpu()
            if name not in sums:
                sums[name] = diff * bs
            else:
                sums[name] += diff * bs

    return {name: value / max(count, 1) for name, value in sums.items()}


def vla_casd(shift_a: dict[str, torch.Tensor], shift_b: dict[str, torch.Tensor]) -> dict[str, float]:
    distances: dict[str, float] = {}
    mean_distances: dict[str, float] = {}
    for name in shift_a:
        abs_diff = (shift_a[name] - shift_b[name]).abs()
        distances[name] = abs_diff.sum().item()
        mean_distances[f"{name}_mean"] = abs_diff.mean().item()
    distances["total"] = sum(distances.values())
    distances.update(mean_distances)
    distances["total_mean"] = sum(mean_distances.values()) / max(len(mean_distances), 1)
    return distances
