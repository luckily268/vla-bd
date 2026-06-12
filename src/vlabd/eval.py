from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_clean(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        pred = model(batch["image"].to(device), batch["tokens"].to(device)).argmax(dim=-1)
        label = batch["clean_label"].to(device)
        correct += (pred == label).sum().item()
        total += label.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_asr(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    success = 0
    total = 0
    for batch in loader:
        pred = model(batch["image"].to(device), batch["tokens"].to(device)).argmax(dim=-1)
        attack_label = batch["attack_label"].to(device)
        success += (pred == attack_label).sum().item()
        total += attack_label.numel()
    return success / max(total, 1)
