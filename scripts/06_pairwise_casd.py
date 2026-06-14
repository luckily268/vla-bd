from __future__ import annotations

import argparse
import json

import torch

from step_utils import load_experiment, make_dataset, make_loader, make_model
from vlabd.casd import average_feature_shift, vla_casd
from vlabd.train import get_device, load_checkpoint


CHECKPOINTS = {
    "remove_lang": "M_remove_lang.pt",
    "remove_vis": "M_remove_vis.pt",
    "remove_both": "M_remove_both.pt",
    "control": "M_control.pt",
}


def flatten_shift(shift: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([shift[name].flatten() for name in sorted(shift)])


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom.item() == 0:
        return 0.0
    return torch.dot(a, b).div(denom).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trigger", default="none")
    args = parser.parse_args()

    cfg, dirs, tokenizer = load_experiment(args.config)
    device = get_device(cfg.get("device", "cuda"))
    train_cfg = cfg["train"]
    batch_size = train_cfg["batch_size"]
    eval_samples = train_cfg["eval_samples"]

    loader = make_loader(
        make_dataset(
            cfg,
            tokenizer,
            "eval",
            eval_samples,
            cfg["seed"] + 7000,
            args.trigger,
        ),
        batch_size,
        shuffle=False,
    )

    bd_model = make_model(cfg, tokenizer.vocab_size)
    load_checkpoint(bd_model, dirs["checkpoints"] / "M_bd.pt", device)

    shifts: dict[str, dict[str, torch.Tensor]] = {}
    flat: dict[str, torch.Tensor] = {}
    for name, ckpt in CHECKPOINTS.items():
        model = make_model(cfg, tokenizer.vocab_size)
        load_checkpoint(model, dirs["checkpoints"] / ckpt, device)
        shift = average_feature_shift(bd_model, model, loader, device)
        shifts[name] = shift
        flat[name] = flatten_shift(shift)

    pairwise: dict[str, dict[str, float]] = {}
    names = list(CHECKPOINTS)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            row = vla_casd(shifts[name_a], shifts[name_b])
            row["cosine"] = cosine_similarity(flat[name_a], flat[name_b])
            pairwise[f"{name_a}_vs_{name_b}"] = row

    result = {
        "trigger_eval": args.trigger,
        "note": "Lower distance and higher cosine mean two removal-induced feature shifts are more similar.",
        "pairwise": pairwise,
    }
    out_path = dirs["metrics"] / f"pairwise_casd_{args.trigger}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
