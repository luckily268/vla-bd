from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    desc: str,
) -> nn.Module:
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        losses = []
        pbar = tqdm(loader, desc=f"{desc} epoch {epoch + 1}/{epochs}", leave=False)
        for batch in pbar:
            image = batch["image"].to(device)
            tokens = batch["tokens"].to(device)
            label = batch["label"].to(device)
            logits = model(image, tokens)
            loss = loss_fn(logits, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses) / len(losses))
    return model


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    torch.save(model.state_dict(), Path(path))


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> nn.Module:
    model.load_state_dict(torch.load(Path(path), map_location=device))
    model.to(device)
    return model
