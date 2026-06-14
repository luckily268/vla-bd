from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlabd.config import ensure_dirs, load_config
from vlabd.data import ToySpec, ToyTokenizer, ToyVLADataset
from vlabd.eval import evaluate_asr, evaluate_clean
from vlabd.model import MiniVLA
from vlabd.train import get_device, save_checkpoint, train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_set(cfg: dict[str, Any], path: str, value: Any) -> None:
    cur = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def make_model(cfg: dict[str, Any], vocab_size: int) -> MiniVLA:
    model_cfg = cfg["model"]
    return MiniVLA(
        vocab_size=vocab_size,
        text_dim=model_cfg["text_dim"],
        vision_dim=model_cfg["vision_dim"],
        fusion_dim=model_cfg["fusion_dim"],
        dropout=model_cfg["dropout"],
    )


def make_dataset(
    cfg: dict[str, Any],
    tokenizer: ToyTokenizer,
    mode: str,
    n: int,
    seed: int,
    eval_trigger: str = "none",
) -> ToyVLADataset:
    train_cfg = cfg["train"]
    trigger_cfg = cfg["triggers"]
    spec = ToySpec(
        image_size=cfg["image_size"],
        visual_patch_size=trigger_cfg["visual_patch_size"],
        lang_words=tuple(trigger_cfg["lang_words"]),
    )
    return ToyVLADataset(
        n=n,
        tokenizer=tokenizer,
        spec=spec,
        mode=mode,
        seed=seed,
        poison_ratio_lang=train_cfg["poison_ratio_lang"],
        poison_ratio_vis=train_cfg["poison_ratio_vis"],
        removal_ratio=train_cfg["removal_ratio"],
        eval_trigger=eval_trigger,
    )


def make_loader(dataset, batch_size: int, shuffle: bool):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate_all(model, cfg, tokenizer, device, seed_offset=1000) -> dict[str, float]:
    train_cfg = cfg["train"]
    batch_size = train_cfg["batch_size"]
    eval_samples = train_cfg["eval_samples"]
    clean = make_loader(
        make_dataset(cfg, tokenizer, "eval", eval_samples, cfg["seed"] + seed_offset, "none"),
        batch_size,
        False,
    )
    lang = make_loader(
        make_dataset(cfg, tokenizer, "eval", eval_samples, cfg["seed"] + seed_offset + 1, "lang"),
        batch_size,
        False,
    )
    vis = make_loader(
        make_dataset(cfg, tokenizer, "eval", eval_samples, cfg["seed"] + seed_offset + 2, "vis"),
        batch_size,
        False,
    )
    return {
        "clean_acc": evaluate_clean(model, clean, device),
        "lang_asr": evaluate_asr(model, lang, device),
        "vis_asr": evaluate_asr(model, vis, device),
    }


def run_one(cfg: dict[str, Any], run_dir: Path, device: torch.device) -> dict[str, Any]:
    cfg["output_dir"] = str(run_dir)
    dirs = ensure_dirs(run_dir)
    tokenizer = ToyTokenizer(cfg["triggers"]["lang_words"])
    train_cfg = cfg["train"]
    set_seed(cfg["seed"])

    clean_loader = make_loader(
        make_dataset(cfg, tokenizer, "clean", train_cfg["clean_samples"], cfg["seed"] + 10),
        train_cfg["batch_size"],
        True,
    )
    poison_loader = make_loader(
        make_dataset(cfg, tokenizer, "poisoned", train_cfg["poison_samples"], cfg["seed"] + 20),
        train_cfg["batch_size"],
        True,
    )

    clean_model = make_model(cfg, tokenizer.vocab_size)
    train_model(
        clean_model,
        clean_loader,
        train_cfg["clean_epochs"],
        train_cfg["learning_rate"],
        train_cfg["weight_decay"],
        device,
        "clean",
        show_progress=False,
    )
    save_checkpoint(clean_model, dirs["checkpoints"] / "M_clean.pt")

    bd_model = copy.deepcopy(clean_model)
    train_model(
        bd_model,
        poison_loader,
        train_cfg["backdoor_epochs"],
        train_cfg["learning_rate"],
        train_cfg["weight_decay"],
        device,
        "backdoor",
        show_progress=False,
    )
    save_checkpoint(bd_model, dirs["checkpoints"] / "M_bd.pt")

    models = {"M_bd": bd_model}
    for model_name, mode, seed_offset in [
        ("M_remove_lang", "remove_lang", 3100),
        ("M_remove_vis", "remove_vis", 3200),
        ("M_control", "clean", 3300),
    ]:
        loader = make_loader(
            make_dataset(cfg, tokenizer, mode, train_cfg["removal_samples"], cfg["seed"] + seed_offset),
            train_cfg["batch_size"],
            True,
        )
        model = copy.deepcopy(bd_model)
        train_model(
            model,
            loader,
            train_cfg["removal_epochs"],
            train_cfg.get("removal_learning_rate", train_cfg["learning_rate"]),
            train_cfg["weight_decay"],
            device,
            model_name,
            show_progress=False,
        )
        save_checkpoint(model, dirs["checkpoints"] / f"{model_name}.pt")
        models[model_name] = model

    metrics = {
        name: evaluate_all(model, cfg, tokenizer, device, seed_offset=2000 + i * 100)
        for i, (name, model) in enumerate(models.items())
    }
    (dirs["metrics"] / "cross_removal_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return summarize_metrics(metrics)


def summarize_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    bd = metrics["M_bd"]
    remove_lang = metrics["M_remove_lang"]
    remove_vis = metrics["M_remove_vis"]
    control = metrics["M_control"]
    return {
        "bd_clean_acc": bd["clean_acc"],
        "bd_lang_asr": bd["lang_asr"],
        "bd_vis_asr": bd["vis_asr"],
        "remove_lang_clean_acc": remove_lang["clean_acc"],
        "remove_lang_lang_asr": remove_lang["lang_asr"],
        "remove_lang_vis_asr": remove_lang["vis_asr"],
        "remove_vis_clean_acc": remove_vis["clean_acc"],
        "remove_vis_lang_asr": remove_vis["lang_asr"],
        "remove_vis_vis_asr": remove_vis["vis_asr"],
        "control_clean_acc": control["clean_acc"],
        "control_lang_asr": control["lang_asr"],
        "control_vis_asr": control["vis_asr"],
        "lang_to_vis_extra_drop": control["vis_asr"] - remove_lang["vis_asr"],
        "vis_to_lang_extra_drop": control["lang_asr"] - remove_vis["lang_asr"],
    }


def build_runs(base_cfg: dict[str, Any], max_kind: str) -> list[tuple[str, dict[str, Any], str, Any]]:
    runs: list[tuple[str, dict[str, Any], str, Any]] = []

    def add(kind: str, name: str, path: str, value: Any) -> None:
        cfg = copy.deepcopy(base_cfg)
        deep_set(cfg, path, value)
        runs.append((f"{kind}_{name}", cfg, path, value))

    for value in [5e-4, 1e-4, 5e-5, 1e-5]:
        add("removal_lr", f"{value:g}", "train.removal_learning_rate", value)

    for value in [0.05, 0.10, 0.25, 0.50]:
        add("removal_ratio", f"{value:g}", "train.removal_ratio", value)

    for value in [1, 2, 5, 10]:
        add("removal_epochs", str(value), "train.removal_epochs", value)

    if max_kind in {"standard", "full"}:
        for value in [0.02, 0.05, 0.10, 0.20]:
            cfg = copy.deepcopy(base_cfg)
            deep_set(cfg, "train.poison_ratio_lang", value)
            deep_set(cfg, "train.poison_ratio_vis", value)
            runs.append((f"poison_ratio_{value:g}", cfg, "train.poison_ratio_lang/vis", value))

    if max_kind == "full":
        for value in [3, 7, 11, 17, 23]:
            add("seed", str(value), "seed", value)

    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/toy_lang_vis_low_lr_removal.json")
    parser.add_argument("--suite", choices=["quick", "standard", "full"], default="standard")
    parser.add_argument("--output-dir", default="outputs/ablations")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = get_device(base_cfg.get("device", "cuda"))
    print(f"Using device: {device}")

    runs = build_runs(base_cfg, args.suite)
    rows: list[dict[str, Any]] = []
    for idx, (run_name, cfg, variable, value) in enumerate(runs, start=1):
        run_dir = out_root / run_name
        print(f"[{idx}/{len(runs)}] {run_name}: {variable}={value}")
        summary = run_one(cfg, run_dir, device)
        row = {
            "run": run_name,
            "variable": variable,
            "value": value,
            **summary,
        }
        rows.append(row)
        print(
            "  "
            f"bd(lang,vis)=({row['bd_lang_asr']:.3f},{row['bd_vis_asr']:.3f}) "
            f"extra_drop(lang->vis,vis->lang)=({row['lang_to_vis_extra_drop']:.3f},"
            f"{row['vis_to_lang_extra_drop']:.3f})"
        )

    json_path = out_root / "summary.json"
    csv_path = out_root / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
