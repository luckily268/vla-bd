from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from step_utils import load_experiment, make_dataset, make_loader, make_model
from vlabd.train import get_device, load_checkpoint, save_checkpoint, train_model


REMOVAL_MODES = {
    "lang": ("M_remove_lang.pt", "remove_lang", 3100),
    "vis": ("M_remove_vis.pt", "remove_vis", 3200),
    "both": ("M_remove_both.pt", "remove_both", 3250),
    "control": ("M_control.pt", "clean", 3300),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", choices=REMOVAL_MODES.keys(), required=True)
    args = parser.parse_args()

    cfg, dirs, tokenizer = load_experiment(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "cuda"))
    train_cfg = cfg["train"]
    output_name, dataset_mode, seed_offset = REMOVAL_MODES[args.target]

    print(f"Using device: {device}")
    print(f"Step 3: train {output_name} from M_bd using mode={dataset_mode}.")

    model = make_model(cfg, tokenizer.vocab_size)
    load_checkpoint(model, dirs["checkpoints"] / "M_bd.pt", device)

    dataset = make_dataset(
        cfg,
        tokenizer,
        mode=dataset_mode,
        n=train_cfg["removal_samples"],
        seed=cfg["seed"] + seed_offset,
    )
    loader = make_loader(dataset, train_cfg["batch_size"], shuffle=True)

    train_model(
        model,
        loader,
        train_cfg["removal_epochs"],
        train_cfg.get("removal_learning_rate", train_cfg["learning_rate"]),
        train_cfg["weight_decay"],
        device,
        args.target,
    )

    out = dirs["checkpoints"] / output_name
    save_checkpoint(model, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
