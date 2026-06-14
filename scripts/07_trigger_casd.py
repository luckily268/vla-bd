from __future__ import annotations

import argparse
import json

import torch

from step_utils import load_experiment, make_dataset, make_loader, make_model
from vlabd.casd import vla_casd
from vlabd.train import get_device, load_checkpoint


TRIGGERS = ["lang", "vis", "both_text", "both_vis", "both"]


def flatten_shift(shift: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([shift[name].flatten() for name in sorted(shift)])


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom.item() == 0:
        return 0.0
    return torch.dot(a, b).div(denom).item()


@torch.no_grad()
def average_trigger_shift(model, clean_loader, trigger_loader, device):
    model.eval()
    sums: dict[str, torch.Tensor] = {}
    count = 0
    for clean_batch, trigger_batch in zip(clean_loader, trigger_loader):
        clean_image = clean_batch["image"].to(device)
        clean_tokens = clean_batch["tokens"].to(device)
        trigger_image = trigger_batch["image"].to(device)
        trigger_tokens = trigger_batch["tokens"].to(device)

        _, clean_feats = model(clean_image, clean_tokens, return_features=True)
        _, trigger_feats = model(trigger_image, trigger_tokens, return_features=True)

        bs = clean_image.shape[0]
        count += bs
        for name in clean_feats:
            diff = (trigger_feats[name] - clean_feats[name]).mean(dim=0).detach().cpu()
            if name not in sums:
                sums[name] = diff * bs
            else:
                sums[name] += diff * bs
    return {name: value / max(count, 1) for name, value in sums.items()}


def shift_norms(shift: dict[str, torch.Tensor]) -> dict[str, float]:
    norms = {f"{name}_l2": torch.linalg.vector_norm(value).item() for name, value in shift.items()}
    norms["total_l2"] = torch.linalg.vector_norm(flatten_shift(shift)).item()
    return norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="M_bd.pt")
    args = parser.parse_args()

    cfg, dirs, tokenizer = load_experiment(args.config)
    device = get_device(cfg.get("device", "cuda"))
    train_cfg = cfg["train"]
    batch_size = train_cfg["batch_size"]
    eval_samples = train_cfg["eval_samples"]
    seed = cfg["seed"] + 8000

    model = make_model(cfg, tokenizer.vocab_size)
    load_checkpoint(model, dirs["checkpoints"] / args.model, device)

    clean_loader = make_loader(
        make_dataset(cfg, tokenizer, "eval", eval_samples, seed, "none"),
        batch_size,
        shuffle=False,
    )

    shifts = {}
    flat = {}
    for trigger in TRIGGERS:
        trigger_loader = make_loader(
            make_dataset(cfg, tokenizer, "eval", eval_samples, seed, trigger),
            batch_size,
            shuffle=False,
        )
        shifts[trigger] = average_trigger_shift(model, clean_loader, trigger_loader, device)
        flat[trigger] = flatten_shift(shifts[trigger])

    pairwise = {}
    for i, a in enumerate(TRIGGERS):
        for b in TRIGGERS[i + 1 :]:
            row = vla_casd(shifts[a], shifts[b])
            row["cosine"] = cosine_similarity(flat[a], flat[b])
            pairwise[f"{a}_vs_{b}"] = row

    result = {
        "model": args.model,
        "note": "Compares trigger-induced feature shifts: features(triggered input) - features(clean paired input).",
        "shift_norms": {name: shift_norms(shift) for name, shift in shifts.items()},
        "pairwise": pairwise,
    }
    out_path = dirs["metrics"] / f"trigger_casd_{args.model.replace('.pt', '')}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
