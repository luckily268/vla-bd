from __future__ import annotations

import argparse
import json

from step_utils import load_experiment, make_dataset, make_loader, make_model
from vlabd.eval import evaluate_asr, evaluate_clean
from vlabd.plotting import save_asr_heatmap, save_metric_table
from vlabd.train import get_device, load_checkpoint


CHECKPOINTS = {
    "M_bd": "M_bd.pt",
    "M_remove_lang": "M_remove_lang.pt",
    "M_remove_vis": "M_remove_vis.pt",
    "M_control": "M_control.pt",
}


def has_both_backdoor(cfg: dict) -> bool:
    train_cfg = cfg["train"]
    trigger_cfg = cfg["triggers"]
    return (
        train_cfg.get("poison_ratio_both", 0.0) > 0
        or bool(trigger_cfg.get("both_lang_words"))
    )


def evaluate_model(model, cfg, tokenizer, device, seed_offset: int) -> dict[str, float]:
    train_cfg = cfg["train"]
    batch_size = train_cfg["batch_size"]
    eval_samples = train_cfg["eval_samples"]

    clean_loader = make_loader(
        make_dataset(cfg, tokenizer, "eval", eval_samples, cfg["seed"] + seed_offset, "none"),
        batch_size,
        shuffle=False,
    )
    lang_loader = make_loader(
        make_dataset(cfg, tokenizer, "eval", eval_samples, cfg["seed"] + seed_offset + 1, "lang"),
        batch_size,
        shuffle=False,
    )
    vis_loader = make_loader(
        make_dataset(cfg, tokenizer, "eval", eval_samples, cfg["seed"] + seed_offset + 2, "vis"),
        batch_size,
        shuffle=False,
    )
    metrics = {
        "clean_acc": evaluate_clean(model, clean_loader, device),
        "lang_asr": evaluate_asr(model, lang_loader, device),
        "vis_asr": evaluate_asr(model, vis_loader, device),
    }
    if has_both_backdoor(cfg):
        both_loader = make_loader(
            make_dataset(cfg, tokenizer, "eval", eval_samples, cfg["seed"] + seed_offset + 3, "both"),
            batch_size,
            shuffle=False,
        )
        both_text_loader = make_loader(
            make_dataset(cfg, tokenizer, "eval", eval_samples, cfg["seed"] + seed_offset + 4, "both_text"),
            batch_size,
            shuffle=False,
        )
        both_vis_loader = make_loader(
            make_dataset(cfg, tokenizer, "eval", eval_samples, cfg["seed"] + seed_offset + 5, "both_vis"),
            batch_size,
            shuffle=False,
        )
        metrics["both_text_only_asr"] = evaluate_asr(model, both_text_loader, device)
        metrics["both_vis_only_asr"] = evaluate_asr(model, both_vis_loader, device)
        metrics["both_asr"] = evaluate_asr(model, both_loader, device)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg, dirs, tokenizer = load_experiment(args.config)
    device = get_device(cfg.get("device", "cuda"))

    print(f"Using device: {device}")
    print("Step 4: evaluate clean accuracy and triggered ASR.")

    metrics = {}
    checkpoints = dict(CHECKPOINTS)
    if has_both_backdoor(cfg):
        checkpoints["M_remove_both"] = "M_remove_both.pt"

    for i, (model_name, ckpt_name) in enumerate(checkpoints.items()):
        model = make_model(cfg, tokenizer.vocab_size)
        load_checkpoint(model, dirs["checkpoints"] / ckpt_name, device)
        metrics[model_name] = evaluate_model(model, cfg, tokenizer, device, 2000 + i * 100)

    metrics_path = dirs["metrics"] / "cross_removal_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_metric_table(metrics, dirs["figures"] / "metric_table.png")
    save_asr_heatmap(metrics, dirs["figures"] / "cross_removal_asr.png")

    print(json.dumps(metrics, indent=2))
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
