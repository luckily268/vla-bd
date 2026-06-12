from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from step_utils import load_experiment, make_dataset, make_loader, make_model
from vlabd.train import get_device, load_checkpoint, save_checkpoint, train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg, dirs, tokenizer = load_experiment(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "cuda"))
    train_cfg = cfg["train"]

    print(f"Using device: {device}")
    print("Step 2: load M_clean and inject language/visual backdoors.")

    model = make_model(cfg, tokenizer.vocab_size)
    load_checkpoint(model, dirs["checkpoints"] / "M_clean.pt", device)

    poison_dataset = make_dataset(
        cfg,
        tokenizer,
        mode="poisoned",
        n=train_cfg["poison_samples"],
        seed=cfg["seed"] + 20,
    )
    poison_loader = make_loader(poison_dataset, train_cfg["batch_size"], shuffle=True)

    train_model(
        model,
        poison_loader,
        train_cfg["backdoor_epochs"],
        train_cfg["learning_rate"],
        train_cfg["weight_decay"],
        device,
        "backdoor",
    )

    out = dirs["checkpoints"] / "M_bd.pt"
    save_checkpoint(model, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
